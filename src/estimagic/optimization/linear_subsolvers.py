"""Collection of linear trust-region subsolvers."""
import math

import numpy as np

ZERO_THRESH = 1e-14


def trsbox_linear(model_gradient, lower_bound, upper_bound, trustregion_radius):
    """Minimize a linear trust-region subproblem using the trsbox algorithm.

    Solve the linear subproblem:

      min_x   g' * x
        s.t.   lower_bound <= x <= upper_bound
              ||x||^2 <= delta^2

    using an active-set approach.

    This algorithm is an implementation of the routine TRSBOX from
        M. J. D. Powell (2009) "The BOBYQA algorithm for bound constrained
        optimization without derivatives."

    Args:
        model_gradient (np.ndarray): Gradient of the Lagrange polynomial.
            Array of shape (n,).
        lower_bound (np.ndarray): Lower bounds for x. Array of shape (n,).
        upper_bound (np.ndarray): Upper bounds for x. Array of shape (n,).
        trustregion_radius (float): Radius of the trust-region.

    Returns:
        (np.ndarray): Solution vector to the linear trust-region subproblem.
            Array of shape (n,).
    """
    lower_bound = np.minimum(lower_bound, -ZERO_THRESH)
    upper_bound = np.maximum(upper_bound, ZERO_THRESH)

    n = model_gradient.shape[0]
    x_candidate = np.zeros(n)

    direction = -model_gradient

    constant_directions = np.where(np.abs(direction) < ZERO_THRESH)[0]
    direction[constant_directions] = 0.0

    active_directions = np.setdiff1d(np.arange(n), constant_directions)
    set_active_directions = iter(active_directions)

    niter = 0
    while niter < n:
        niter += 1

        if np.linalg.norm(direction) < ZERO_THRESH:
            break

        x_candidate_unconstr = _take_unconstrained_step_up_to_boundary(
            x_candidate, direction, trustregion_radius
        )

        active_bound, index_active_bound = _find_next_active_bound(
            x_candidate_unconstr,
            lower_bound,
            upper_bound,
            set_active_directions,
        )

        if active_bound is None:
            x_candidate = x_candidate_unconstr
            break

        else:
            x_candidate, direction = _take_constrained_step_up_to_boundary(
                x_candidate,
                direction,
                active_bound,
                index_active_bound,
            )

    return x_candidate


def trsbox_geometry(
    x_initial,
    model_gradient,
    constant_term,
    lower_bound,
    upper_bound,
    trustregion_radius,
):
    """Maximize a Lagrange polynomial of degree one.

    Given a Lagrange polynomial defined by
        L(x) = c + g' * (x - x_initial)

    maximize abs(L(x)) in a trust region setting, i.e. solve:
        max_x  abs(c + g' * (x - x_initial))
            s.t. lower_bound <= x <= upper_bound
                 ||x - x_initial|| <= delta


    To find the solution, g' * (x - x_initial) is both minimized and maximized.
    The resulting candidate vectors are then plugged into the objective function L(x)
    to see which one yields the largest absolute value.

    Args:
        x_initial (np.ndarray): Initial candidate vector x of shape (n,).
            The solution to L(x) is defined as  the step x* = x_accepted - x_base.
        model_gradient (np.ndarray): Gradient of the Lagrange polynomial.
            Array of shape (n,).
        constant_term (float): Constant term of the Lagrange polynomial.
        lower_bound (np.ndarray): Lower bounds for x. Array of shape (n,).
        upper_bound (np.ndarray): Upper bounds for x. Array of shape (n,).
        trustregion_radius (float): Radius of the trust-region.

    Returns:
        (np.ndarray): Solution vector of shape (n,) that maximizes the Lagrange
            polynomial.
    """
    # Check if bounds valid
    if np.any(lower_bound > x_initial + ZERO_THRESH):
        raise ValueError("x_base violates lower bound.")
    if np.any(x_initial - ZERO_THRESH > upper_bound):
        raise ValueError("x_base violates upper bound.")

    # Minimize and maximize g' * (x - x_initial), respectively
    x_candidate_min = trsbox_linear(
        model_gradient,
        lower_bound - x_initial,
        upper_bound - x_initial,
        trustregion_radius,
    )
    x_candidate_max = trsbox_linear(
        -model_gradient,
        lower_bound - x_initial,
        upper_bound - x_initial,
        trustregion_radius,
    )

    lagrange_polynomial = lambda x: abs(constant_term + np.dot(model_gradient, x))

    if lagrange_polynomial(x_candidate_min) >= lagrange_polynomial(x_candidate_max):
        x_accepted = x_candidate_min + x_initial
    else:
        x_accepted = x_candidate_max + x_initial

    return x_accepted


def _find_next_active_bound(
    x_candidate_unconstr,
    lower_bound,
    upper_bound,
    set_active_directions,
):
    """Find the next active bound and return its index.

    A (lower or upper) bound is considered active if
        x_candidate <= lower_bound
        x_candidate >= upper_bound

    Args:
        x_candidate_unconstr (np.ndarray): Unconstrained candidate vector of shape (n,),
            which has been computed without taking bounds into account.
        lower_bound (np.ndarray): Lower bounds for x. Array of shape (n,).
        upper_bound (np.ndarray): Upper bounds for x. Array of shape (n,).
        set_active_directions (iterator): Iterator over the indices of active search
            directions, i.e. directions that are not zero.

    Returns:
        (tuple):
            - active_bound (float or None): The next active bound. It can be a lower
                or active bound. If None, there are no more active bounds left in the
                set of active search directions.
            - index_bound_active (int or None): Index where an active lower or
                upper bound has been found. None, if no active bound has been detected.
    """
    index_active = next(set_active_directions)

    while True:
        if x_candidate_unconstr[index_active] >= upper_bound[index_active]:
            active_bound = upper_bound[index_active]
            break

        elif x_candidate_unconstr[index_active] <= lower_bound[index_active]:
            active_bound = lower_bound[index_active]
            break

        else:
            try:
                index_active = next(set_active_directions)
            except StopIteration:
                active_bound = None
                break

    return active_bound, index_active


def _take_constrained_step_up_to_boundary(
    x_candidate, direction, active_bound, index_bound_active
):
    """Take largest constrained step possible until trust-region boundary is hit.

    Args:
        x_candidate (np.ndarray): Current candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        lower_bound (np.ndarray): Lower bounds for x. Array of shape (n,).
        upper_bound (np.ndarray): Upper bounds for x. Array of shape (n,).
        index_bound_active (int): Index where an active lower or upper bound
            has been found.
        upper_bound_hit (bool): True, if the active bound detected is an
            upper bound. False, if the active bound is a lower bound.

    Returns:
        (tuple):
        - x_candidate (np.ndarray): New candidate vector of shape (n,).
        - direction (np.ndarray): New direction vector of shape (n,), where the
            search direction of the currently active bound has been set to zero.
    """
    step_size_constr = (active_bound - x_candidate[index_bound_active]) / direction[
        index_bound_active
    ]

    x_candidate = x_candidate + step_size_constr * direction
    x_candidate[index_bound_active] = active_bound

    # Do not search in this direction anymore
    direction[index_bound_active] = 0.0

    return x_candidate, direction


def _take_unconstrained_step_up_to_boundary(x_candidate, direction, trustregion_radius):
    """Take largest unconstrained step possible until trust-region boundary is hit.

    Args:
        x_candidate (np.ndarray): Current candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        trustregion_radius (float): Radius of the trust-region.

    Returns:
        (np.ndarray): New unconstrained candidate vector shape (n,).
    """
    step_size_unconstr = _get_distance_to_trustregion_boundary(
        x_candidate, direction, trustregion_radius
    )
    x_candidate_unconstr = x_candidate + step_size_unconstr * direction

    return x_candidate_unconstr


def _get_distance_to_trustregion_boundary(x0, direction, radius):
    """Compute the candidate vector's distance to the trustregion boundary.

    Given the candidate vector, find the largest step `alpha` in direction `g`
    that satisfies ||x|| <= delta,

    where `g` denotes the direction vector and `delta` the trust-region radius.

    To find `alpha`, i.e. the candidate's distance to the trust-region boundary, solve
      ||x0 + alpha * g||^2 = delta^2
         s.t. alpha >= 0

    Using this method, the solution exists whenever ||x0|| <= delta^2.

    Choose alpha = 0, if the direction vector is zero everywhere.

    Args:
        x0 (np.ndarray): Candidate vector of shape (n,).
        direction (np.ndarray): Direction vector of shape (n,).
        radius (float): Radius of the trust-region.

    Returns:
        (float) Distance of the candidate vector to the trustregion
            boundary.
    """
    g_dot_x0 = np.dot(direction, x0)
    g_sumsq = np.dot(direction, direction)
    x0_sumsq = np.dot(x0, x0)

    if math.sqrt(g_sumsq) < ZERO_THRESH:
        alpha = 0
    else:
        alpha = (
            math.sqrt(np.maximum(0, g_dot_x0**2 + g_sumsq * (radius**2 - x0_sumsq)))
            - g_dot_x0
        ) / g_sumsq

    return alpha

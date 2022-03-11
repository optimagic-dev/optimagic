from copy import copy
from functools import reduce

import numpy as np
from estimagic.optimization.trustregion_conjugate_gradient import minimize_trust_cg

EPSILON = np.finfo(float).eps ** (2 / 3)


def minimize_trust_bntr(
    x_candidate,
    linear_terms,
    square_terms,
    lower_bound,
    upper_bound,
    options,
):
    """Minimize a bounded trust-region subproblem via Newton Conjugate Gradient."""
    if "gatol" not in options.keys():
        options["gatol"] = 1e-8
    if "grtol" not in options.keys():
        options["grtol"] = 1e-8

    (
        x_candidate,
        f_candidate,
        model_gradient,
        model_hessian,
        trustregion_radius,
        active_bounds,
        inactive_bounds,
        converged,
    ) = take_preliminary_steepest_descent_step_and_check_for_solution(
        x_candidate, linear_terms, square_terms, lower_bound, upper_bound, options
    )

    niter = 0

    while converged is False:
        niter += 1

        x_old = np.copy(x_candidate)
        f_old = copy(f_candidate)

        accept_step = False

        while accept_step is False and converged is False:
            (
                direction_inactive,
                conjugate_gradient_step,
                model_gradient,
                gradient_norm,
            ) = compute_conjugate_gradient_step(
                x_candidate,
                model_gradient,
                model_hessian,
                lower_bound,
                upper_bound,
                active_bounds,
                inactive_bounds,
                trustregion_radius,
            )

            # Temporarily accept the step and project it into the bounds
            x_candidate = x_candidate - conjugate_gradient_step
            x_candidate = apply_bounds_to_x_candidate(
                x_candidate, lower_bound, upper_bound
            )
            f_candidate = _compute_criterion_main_model(
                x_candidate, model_gradient["initial"], model_hessian["initial"]
            )

            predicted_reduction = compute_predicted_reduction(
                direction_inactive,
                conjugate_gradient_step,
                model_gradient,
                model_hessian,
                active_bounds,
                inactive_bounds,
            )

            actual_reduction = f_old - f_candidate

            trustregion_radius_old = copy(trustregion_radius)

            trustregion_radius, accept_step = update_trustregion_radius(
                predicted_reduction,
                actual_reduction,
                gradient_norm,
                trustregion_radius,
                f_candidate,
            )

            if accept_step:
                model_gradient["candidate"] = model_gradient["initial"] + np.dot(
                    model_hessian["initial"], x_candidate
                )

                model_gradient["bounds_inactive"] = np.copy(model_gradient["candidate"])
                model_gradient["bounds_inactive"][active_bounds["all"]] = 0

            else:
                x_candidate = np.copy(x_old)
                f_candidate = np.copy(f_old)

                if trustregion_radius_old == trustregion_radius:
                    break

            converged = _check_for_convergence(
                x_candidate,
                model_gradient,
                f_candidate,
                lower_bound,
                upper_bound,
                options,
            )

        if converged is True:
            break

    return x_candidate, niter


def compute_predicted_reduction(
    direction_inactive,
    conjugate_gradient_step,
    model_gradient,
    model_hessian,
    active_bounds,
    inactive_bounds,
):
    """Compute predicted reduction in f induced by current Conjugate Gradient step."""
    if active_bounds["all"].size > 0:
        # Projection changed the step, so we have to recompute the step
        # and the predicted reduction. Leave the rust radius unchanged.
        direction_inactive = conjugate_gradient_step[inactive_bounds]
        gradient_inactive = model_gradient["candidate"][inactive_bounds]

        predicted_reduction = _compute_criterion_main_model(
            direction_inactive, gradient_inactive, model_hessian["bounds_inactive"]
        )
    else:
        # Step did not change, so we can just recover the
        # pre-computed prediction
        predicted_reduction = _compute_criterion_main_model(
            direction_inactive,
            model_gradient["bounds_inactive"],
            model_hessian["bounds_inactive"],
        )

        predicted_reduction = -predicted_reduction

    return predicted_reduction


def update_trustregion_radius(
    predicted_reduction, actual_reduction, dnorm, tr_radius, f_candidate
):
    """Update the trust-region radius based on predicted and actual reduction."""
    eta1 = 1.0e-4
    eta2 = 0.25
    eta3 = 0.50
    eta4 = 0.90

    alpha1 = 0.25
    alpha2 = 0.50
    alpha3 = 1.00
    alpha4 = 2.00
    alpha5 = 4.00

    accept_step = False

    if predicted_reduction < 0 or np.isfinite(predicted_reduction) is False:
        # The predicted reduction has the wrong sign.
        # Reject the just peformed Conjuate Gradient step and start over.
        tr_radius = alpha1 * min(tr_radius, dnorm)
    else:
        if np.isfinite(actual_reduction) is False:
            tr_radius = alpha1 * min(tr_radius, dnorm)
        else:
            if abs(actual_reduction) <= max(1, abs(f_candidate) * EPSILON) and abs(
                predicted_reduction
            ) <= max(1, abs(f_candidate) * EPSILON):
                kappa = 1
            else:
                kappa = actual_reduction / predicted_reduction

            if kappa < eta1:
                tr_radius = alpha1 * min(tr_radius, dnorm)
            else:
                accept_step = True

                # Update the trust-region radius only if the computed step is at the
                # trust radius boundary
                if np.allclose(dnorm, tr_radius):
                    if kappa < eta2:
                        # Marginal bad step
                        tr_radius = alpha2 * tr_radius
                    elif kappa < eta3:
                        # Reasonable step
                        tr_radius = alpha3 * tr_radius
                    elif kappa < eta4:
                        tr_radius = alpha4 * tr_radius
                    else:
                        # Very good step
                        tr_radius = alpha5 * tr_radius

    tr_radius = max(min(tr_radius, 1e10), 1e-10)

    return tr_radius, accept_step


def compute_conjugate_gradient_step(
    x,
    model_gradient,
    model_hessian,
    lower_bound,
    upper_bound,
    active_set,
    inactive_set,
    trustregion_radius,
):
    """Compute the bounded Conjugate Gradient trust-region step."""
    model_gradient_updated = model_gradient.copy()

    n = x.shape[0]
    conjugate_gradient_step = np.zeros(n)

    model_gradient_updated["bounds_inactive"] = model_gradient["candidate"][
        inactive_set
    ]

    direction_inactive = minimize_trust_cg(
        model_gradient_updated["bounds_inactive"],
        model_hessian["bounds_inactive"],
        trustregion_radius,
    )
    gradient_norm = np.linalg.norm(direction_inactive)

    conjugate_gradient_step[inactive_set] = -direction_inactive
    conjugate_gradient_step = _apply_bounds_to_conjugate_gradient_step(
        x,
        lower_bound,
        upper_bound,
        active_set,
        conjugate_gradient_step,
    )

    return (
        direction_inactive,
        conjugate_gradient_step,
        model_gradient_updated,
        gradient_norm,
    )


def _apply_bounds_to_conjugate_gradient_step(
    x,
    lower_bound,
    upper_bound,
    active_set,
    cg_step_direction=None,
):
    """Apply lower and upper bounds to the Conjugate Gradient step direction."""
    n = x.shape[0]

    if cg_step_direction is None:
        cg_step_direction = np.zeros(n)

    indices_active_fixed = active_set["fixed"]
    indices_active_lower = active_set["lower"]
    indices_active_upper = active_set["upper"]

    if indices_active_lower.size > 0:
        x_active_lower = x[indices_active_lower]
        lower_bound_active = lower_bound[indices_active_lower]

        cg_step_direction[indices_active_lower] = lower_bound_active - x_active_lower

    if indices_active_upper.size > 0:
        x_active_upper = x[indices_active_upper]
        upper_bound_active = upper_bound[indices_active_upper]

        cg_step_direction[indices_active_upper] = upper_bound_active - x_active_upper

    if indices_active_fixed.size > 0:
        cg_step_direction[indices_active_fixed] = 0

    return cg_step_direction


def take_preliminary_steepest_descent_step_and_check_for_solution(
    x_candidate,
    linear_terms,
    square_terms,
    lower_bound,
    upper_bound,
    options,
):
    """Take a preliminary steepest descent step and check if it finds a solution."""
    if "gatol" not in options.keys():
        options["gatol"] = 1e-8
    if "grtol" not in options.keys():
        options["grtol"] = 1e-8

    model_gradient = {"initial": linear_terms}
    model_hessian = {"initial": square_terms}

    f_candidate = _compute_criterion_main_model(
        x_candidate, model_gradient["initial"], model_hessian["initial"]
    )

    newton_step = compute_newton_step(model_gradient, model_hessian)

    active_bounds = get_active_bounds(
        x_candidate, lower_bound, upper_bound, model_gradient["initial"], newton_step
    )
    inactive_bounds = get_inactive_bounds(x_candidate, active_bounds)

    model_gradient = find_gradient_subvector_where_bounds_inactive(
        x_candidate, model_gradient, model_hessian, active_bounds["all"]
    )

    converged = _check_for_convergence(
        x_candidate,
        model_gradient,
        f_candidate,
        lower_bound,
        upper_bound,
        options,
    )

    if converged is True:
        x_accepted = x_candidate
        f_accepted = f_candidate
        trustregion_radius = 100
    else:
        model_hessian = find_hessian_submatrix_where_bounds_inactive(
            model_hessian, inactive_bounds
        )

        (
            x_candidate_steepest_descent,
            f_min_steepest_descent,
            trustregion_radius,
            max_trustregion_radius,
        ) = compute_steepest_descent_and_update_trustregion_radius(
            x_candidate,
            f_candidate,
            model_gradient,
            model_hessian,
            lower_bound,
            upper_bound,
            inactive_bounds,
        )

        if f_min_steepest_descent < f_candidate:
            (
                x_accepted,
                f_accepted,
                converged,
            ) = accept_candidate_from_steepest_descent_as_solution(
                x_candidate_steepest_descent,
                f_min_steepest_descent,
                model_gradient,
                model_hessian,
                lower_bound,
                upper_bound,
                active_bounds,
                options,
            )
        else:
            x_accepted = x_candidate
            f_accepted = f_candidate

        trustregion_radius = max(trustregion_radius, max_trustregion_radius)
        trustregion_radius = min(max(trustregion_radius, 1e-10), 1e10)

    return (
        x_accepted,
        f_accepted,
        model_gradient,
        model_hessian,
        trustregion_radius,
        active_bounds,
        inactive_bounds,
        converged,
    )


def compute_newton_step(model_gradient, model_hessian):
    """Compute the newton step."""
    n = model_gradient["initial"].shape[0]
    diag_min = np.ones(n) * 1e-6
    diag_max = np.ones(n) * 1e6

    diag = np.abs(np.diag(model_hessian["initial"]))
    diag_bounded = np.median(np.stack([diag_min, diag, diag_max]), axis=0)

    newton_step = model_gradient["initial"] / diag_bounded

    return newton_step


def get_active_bounds(x, lower_bound, upper_bound, gradient, newton_step):
    """Return the index set of active bounds."""
    step_length = 1e-3

    x_candidate = np.copy(x)
    x_candidate = x_candidate - step_length * newton_step
    x_candidate_bounded = apply_bounds_to_x_candidate(
        x_candidate, lower_bound, upper_bound
    )

    centered_x = x - x_candidate_bounded

    active_fixed = np.where(lower_bound == upper_bound)[0]
    active_lower = np.where((centered_x <= lower_bound) & (gradient > 0))[0]
    active_upper = np.where((centered_x >= upper_bound) & (gradient < 0))[0]
    active_all = reduce(np.union1d, (active_fixed, active_lower, active_upper))

    active_set = {
        "fixed": active_fixed,
        "lower": active_lower,
        "upper": active_upper,
        "all": active_all,
    }

    return active_set


def get_inactive_bounds(x, active_bounds):
    """Return the index set of inactive bounds."""
    entire_set = np.arange(x.shape[0])
    inactive_set = np.setdiff1d(entire_set, active_bounds["all"])

    return inactive_set


def compute_steepest_descent_and_update_trustregion_radius(
    x_candidate,
    f_candidate,
    model_gradient,
    model_hessian,
    lower_bound,
    upper_bound,
    inactive_bounds,
):
    """Perform steepest descent step and update trust-region radius."""
    x_trial = np.copy(x_candidate)
    f_min = np.copy(f_candidate)

    step_size_accepted = 0
    trustregion_radius = 100
    max_trustregion_radius = 0

    gradient_norm = np.linalg.norm(model_gradient["bounds_inactive"])

    for _ in range(5):
        x_old = np.copy(x_trial)

        # Steepest descent step
        step_size_trial = trustregion_radius / gradient_norm
        x_trial = x_trial - step_size_trial * model_gradient["bounds_inactive"]
        x_trial = apply_bounds_to_x_candidate(x_trial, lower_bound, upper_bound)
        f_trial = _compute_criterion_main_model(
            x_trial, model_gradient["initial"], model_hessian["initial"]
        )

        x_accepted = x_trial - x_old

        if f_trial < f_min:
            f_min = copy(f_trial)
            step_size_accepted = trustregion_radius / gradient_norm

        x_inactive = x_accepted[inactive_bounds]

        square_terms = np.dot(
            np.dot(x_inactive, model_hessian["bounds_inactive"]), x_inactive
        )

        predicted_reduction = trustregion_radius * (
            gradient_norm
            - 0.5 * trustregion_radius * square_terms / (gradient_norm**2)
        )
        actual_reduction = f_candidate - f_trial

        tau, max_trustregion_radius = update_tau_and_max_trustregion_radius(
            trustregion_radius,
            predicted_reduction,
            actual_reduction,
            gradient_norm,
            max_trustregion_radius,
        )

        trustregion_radius = tau * trustregion_radius

    x_candidate = x_candidate - step_size_accepted * model_gradient["bounds_inactive"]
    x_candidate = apply_bounds_to_x_candidate(x_candidate, lower_bound, upper_bound)

    return (
        x_candidate,
        f_min,
        trustregion_radius,
        max_trustregion_radius,
    )


def accept_candidate_from_steepest_descent_as_solution(
    x_candidate_steepest_descent,
    f_min_steepest_descent,
    model_gradient,
    model_hessian,
    lower_bound,
    upper_bound,
    active_bounds,
    options,
):
    """Accept the candidate vector from steepest descent step as new solution."""
    newton_step = compute_newton_step(model_gradient, model_hessian)

    active_bounds = get_active_bounds(
        x_candidate_steepest_descent,
        lower_bound,
        upper_bound,
        model_gradient,
        newton_step,
    )

    model_gradient = find_gradient_subvector_where_bounds_inactive(
        x_candidate_steepest_descent,
        model_gradient,
        model_hessian,
        active_bounds["all"],
    )

    converged = _check_for_convergence(
        x_candidate_steepest_descent,
        f_min_steepest_descent,
        lower_bound,
        upper_bound,
        options,
    )

    return x_candidate_steepest_descent, f_min_steepest_descent, converged


def update_tau_and_max_trustregion_radius(
    trustregion_radius, prered, actred, gnorm, max_radius
):
    """Update tau and the new maximum for the trust-region radius."""
    mu1 = 0.35
    mu2 = 0.50
    gamma1 = 0.0625
    gamma2 = 0.5
    gamma3 = 2.0
    gamma4 = 5.0
    theta = 0
    epsilon = np.finfo(float).eps ** (2 / 3)

    if abs(actred) <= epsilon and abs(prered) <= epsilon:
        kappa = 1
    else:
        kappa = actred / prered

    tau_1 = (
        theta
        * gnorm
        * trustregion_radius
        / (theta * gnorm * trustregion_radius + (1 - theta) * prered - actred)
    )
    tau_2 = (
        theta
        * gnorm
        * trustregion_radius
        / (theta * gnorm * trustregion_radius - (1 + theta) * prered + actred)
    )

    tau_min = min(tau_1, tau_2)
    tau_max = max(tau_1, tau_2)

    if abs(kappa - 1) <= mu1:
        # Great agreement
        max_radius = max(max_radius, trustregion_radius)

        if tau_max < 1:
            tau = gamma3
        elif tau_max > gamma4:
            tau = gamma4
        else:
            tau = tau_max

    elif abs(kappa - 1) <= mu2:
        # Good agreement
        max_radius = max(max_radius, trustregion_radius)

        if tau_max < gamma2:
            tau = gamma2
        elif tau_max > gamma3:
            tau = gamma3
        else:
            tau = tau_max

    else:
        # Not good agreement
        if tau_min > 1:
            tau = gamma2
        elif tau_max < gamma1:
            tau = gamma1
        elif (tau_min < gamma1) and (tau_max >= 1):
            tau = gamma1
        elif (
            (tau_1 >= gamma1) and (tau_1 < 1.0) and ((tau_2 < gamma1) or (tau_2 >= 1.0))
        ):
            tau = tau_1
        elif (
            (tau_2 >= gamma1) and (tau_2 < 1.0) and ((tau_1 < gamma1) or (tau_2 >= 1.0))
        ):
            tau = tau_2
        else:
            tau = tau_max

    return tau, max_radius


def find_gradient_subvector_where_bounds_inactive(
    x_candidate, model_gradient, model_hessian, indices_bounds_active
):
    """Find the subvector of the initial gradient where bounds are inactive."""
    model_gradient_updated = model_gradient.copy()

    model_gradient_updated["candidate"] = _compute_derivative_main_model(
        x_candidate, model_gradient["initial"], model_hessian["initial"]
    )

    model_gradient_updated["bounds_inactive"] = np.copy(
        model_gradient_updated["candidate"]
    )
    model_gradient_updated["bounds_inactive"][indices_bounds_active] = 0

    return model_gradient_updated


def find_hessian_submatrix_where_bounds_inactive(model_hessian, inactive_bounds):
    """Find the submatrix of the initial hessian where bounds are inactive."""
    model_hessian_updated = model_hessian.copy()

    model_hessian_updated["bounds_inactive"] = model_hessian["initial"][
        inactive_bounds[:, np.newaxis], inactive_bounds
    ]

    return model_hessian_updated


def _check_for_convergence(
    x_candidate, model_gradient, f_candidate, lower_bound, upper_bound, options
):
    """Check if we have found a solution."""
    direction_fischer_burmeister = get_fischer_burmeister_direction_vector(
        x_candidate, model_gradient["candidate"], lower_bound, upper_bound
    )
    gnorm_fischer = np.linalg.norm(direction_fischer_burmeister)

    if gnorm_fischer < options["gatol"] or (
        f_candidate != 0 and abs(gnorm_fischer / f_candidate) < options["grtol"]
    ):
        converged = True
    else:
        converged = False

    return converged


def apply_bounds_to_x_candidate(x, lower_bound, upper_bound, bound_tol=0):
    """Apply upper and lower bounds to the candidate vector."""
    x = np.where(x <= lower_bound + bound_tol, lower_bound, x)
    x = np.where(x >= upper_bound - bound_tol, upper_bound, x)

    return x


def get_fischer_burmeister_direction_vector(x, gradient, lower_bound, upper_bound):
    """Compute the constrained direction vector via the Fischer-Burmeister function."""
    fischer_vec = np.vectorize(_get_fischer_burmeister_scalar)

    fischer_burmeister = reduce(
        fischer_vec, (upper_bound - x, -gradient, x - lower_bound)
    )
    direction = np.where(
        lower_bound == upper_bound, lower_bound - x, fischer_burmeister
    )

    return direction


def _compute_criterion_main_model(
    x,
    gradient,
    hessian,
):
    """Evaluate the criterion function value of the main model.

    Args:
        x (np.ndarray): Parameter vector of shape (n,).
        model_gradient (np.ndarray): Gradient of the main model of shape (n,).
        model_hessian (np.ndarray): Hessian of the main model of shape (n, n).

    Returns:
        (float): Criterion value of the main model.
    """
    return np.dot(gradient, x) + 0.5 * np.dot(np.dot(x, hessian), x)


def _compute_derivative_main_model(x, model_gradient, model_hessian):
    """Evaluate the derivative of the main model.

    Args:
        model_gradient (np.ndarray): Gradient of the main model of shape (n,).
        model_hessian (np.ndarray): Hessian of the main model of shape (n, n).

    Returns:
        (np.ndarray): derivative of the main model of shape (n,).
    """
    return model_gradient + np.dot(model_hessian, x)


def _get_fischer_burmeister_scalar(a, b):
    """Get the value of the Fischer-Burmeister function for two scalar inputs.

    This method was suggested by Bob Vanderbei. Since the Fischer-Burmeister
    is symmetric, the order of the scalar inputs does not matter.

    Args:
        a (float): First input.
        b (float): Second input.

    Returns:
        float: Value of the Fischer-Burmeister function for inputs a and b.
    """
    if a + b <= 0:
        fischer_burmeister = np.sqrt(a**2 + b**2) - (a + b)
    else:
        fischer_burmeister = -2 * a * b / (np.sqrt(a**2 + b**2) + (a + b))

    return fischer_burmeister

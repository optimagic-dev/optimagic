from collections import namedtuple
from copy import copy
from functools import reduce

import numpy as np
from estimagic.optimization.trustregion_conjugate_gradient import minimize_trust_cg

EPSILON = np.finfo(float).eps ** (2 / 3)


def minimize_bntr_quadratic(
    x_candidate,
    model,
    lower_bound,
    upper_bound,
    options,
):
    """Minimize a bounded trust-region subproblem via Newton Conjugate Gradient.

    https://petsc.org/release/docs/manual/manual.pdf 150, 157, 162
    """
    if "gatol" not in options.keys():
        options["gatol"] = 1e-8
    if "grtol" not in options.keys():
        options["grtol"] = 1e-8
    if "gttol" not in options.keys():
        options["gttol"] = 1e-8
    if "ftol" not in options.keys():
        options["ftol"] = 1e-8
    if "xtol" not in options.keys():
        options["xtol"] = 1e-8
    if "steptol" not in options.keys():
        options["steptol"] = 1e-8
    if "maxiter" not in options.keys():
        options["maxiter"] = 20
    (
        x_candidate,
        f_candidate,
        gradient_candidate,
        hessian_inactive,
        trustregion_radius,
        active_bounds_info,
        converged,
    ) = take_preliminary_steepest_descent_step_and_check_for_solution(
        x_candidate, model, lower_bound, upper_bound, options
    )

    for _niter in range(options["maxiter"]):
        x_old = np.copy(x_candidate)
        f_old = copy(f_candidate)

        accept_step = False

        while accept_step is False and converged is False:
            (
                x_inactive_conjugate_gradient,
                gradient_inactive_conjugate_gradient,
                x_norm_conjugate_gradient,
                conjugate_gradient_step,
            ) = compute_conjugate_gradient_step(
                x_candidate,
                gradient_candidate,
                hessian_inactive,
                lower_bound,
                upper_bound,
                active_bounds_info,
                trustregion_radius,
            )

            # Temporarily accept the step and project it into the bounds
            x_candidate = x_candidate - conjugate_gradient_step
            x_candidate = apply_bounds_to_x_candidate(
                x_candidate, lower_bound, upper_bound
            )
            f_candidate = evaluate_model_criterion(
                x_candidate, model.linear_terms, model.square_terms
            )

            predicted_reduction = (
                compute_predicted_reduction_from_conjugate_gradient_step(
                    x_inactive_conjugate_gradient,
                    gradient_inactive_conjugate_gradient,
                    conjugate_gradient_step,
                    gradient_candidate,
                    hessian_inactive,
                    active_bounds_info,
                )
            )

            actual_reduction = f_old - f_candidate

            trustregion_radius_old = copy(trustregion_radius)

            (
                trustregion_radius,
                accept_step,
            ) = update_trustregion_radius_conjugate_gradient(
                f_candidate,
                predicted_reduction,
                actual_reduction,
                x_norm_conjugate_gradient,
                trustregion_radius,
            )

            if accept_step:
                gradient_candidate = evaluate_model_gradient(x_candidate, model)
            else:
                x_candidate = np.copy(x_old)
                f_candidate = np.copy(f_old)

                if trustregion_radius_old == trustregion_radius:
                    break

            converged = check_for_convergence_conjugate_gradient(
                x_candidate,
                f_candidate,
                x_old,
                f_old,
                gradient_candidate,
                model,
                lower_bound,
                upper_bound,
                trustregion_radius,
                options,
            )

        if converged is True:
            break

    result = {
        "x": x_candidate,
        "criterion": f_candidate,
        "n_iterations": _niter,
        "success": converged,
    }

    return result


def take_preliminary_steepest_descent_step_and_check_for_solution(
    x_candidate,
    model,
    lower_bound,
    upper_bound,
    options,
):
    """Take a preliminary steepest descent step and check if it finds a solution."""
    if "gatol" not in options.keys():
        options["gatol"] = 1e-8
    if "grtol" not in options.keys():
        options["grtol"] = 1e-8
    if "gtol" not in options.keys():
        options["gtol"] = 1e-8

    f_candidate = evaluate_model_criterion(
        x_candidate, model.linear_terms, model.square_terms
    )

    newton_step = compute_newton_step(model)

    active_bounds_info = get_information_on_active_bounds(
        x_candidate, lower_bound, upper_bound, model, newton_step
    )

    gradient_candidate = evaluate_model_gradient(x_candidate, model)
    gradient_projected = np.where(
        np.arange(x_candidate.shape[0]) == active_bounds_info.all, 0, gradient_candidate
    )

    converged = check_for_convergence_steepest_descent(
        x_candidate,
        f_candidate,
        gradient_candidate,
        lower_bound,
        upper_bound,
        options,
    )

    if converged is True:
        x_accepted = x_candidate
        f_accepted = f_candidate

        hessian_inactive = np.copy(model.square_terms)
        trustregion_radius = 100
    else:
        hessian_inactive = find_hessian_submatrix_where_bounds_inactive(
            model, active_bounds_info
        )

        (
            x_candidate_steepest_descent,
            f_min_steepest_descent,
            trustregion_radius,
            max_trustregion_radius,
        ) = perform_steepest_descent_and_update_trustregion_radius(
            x_candidate,
            f_candidate,
            gradient_projected,
            hessian_inactive,
            model,
            lower_bound,
            upper_bound,
            active_bounds_info,
        )

        if f_min_steepest_descent < f_candidate:
            x_accepted = x_candidate_steepest_descent
            f_accepted = f_min_steepest_descent

            gradient_candidate_steepest_descent = evaluate_model_gradient(
                x_candidate_steepest_descent, model
            )

            converged = check_for_convergence_steepest_descent(
                x_candidate_steepest_descent,
                f_min_steepest_descent,
                gradient_candidate_steepest_descent,
                lower_bound,
                upper_bound,
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
        gradient_candidate,
        hessian_inactive,
        trustregion_radius,
        active_bounds_info,
        converged,
    )


def compute_conjugate_gradient_step(
    x,
    gradient_candidate,
    hessian_inactive,
    lower_bound,
    upper_bound,
    active_bounds_info,
    trustregion_radius,
):
    """Compute the bounded Conjugate Gradient trust-region step."""
    conjugate_gradient_step = np.zeros(x.shape[0])
    gradient_inactive_cg = np.copy(gradient_candidate[active_bounds_info.inactive])

    x_inactive_cg = minimize_trust_cg(
        gradient_inactive_cg,
        hessian_inactive,
        trustregion_radius,
    )
    gradient_norm_cg = np.linalg.norm(x_inactive_cg)

    conjugate_gradient_step[active_bounds_info.inactive] = -np.copy(x_inactive_cg)
    conjugate_gradient_step = _apply_bounds_to_conjugate_gradient_step(
        x,
        lower_bound,
        upper_bound,
        active_bounds_info,
        conjugate_gradient_step,
    )

    return (
        x_inactive_cg,
        gradient_inactive_cg,
        gradient_norm_cg,
        conjugate_gradient_step,
    )


def compute_predicted_reduction_from_conjugate_gradient_step(
    x_inactive_cg,
    gradient_inactive_cg,
    conjugate_gradient_step,
    gradient_candidate,
    hessian_inactive,
    active_bounds_info,
):
    """Compute predicted reduction induced by the Conjugate Gradient step."""
    if active_bounds_info.all.size > 0:
        # Projection changed the step, so we have to recompute the step
        # and the predicted reduction. Leave the rust radius unchanged.
        x_inactive_recomp = conjugate_gradient_step[active_bounds_info.inactive]
        gradient_inactive_recomp = gradient_candidate[active_bounds_info.inactive]

        predicted_reduction = evaluate_model_criterion(
            x_inactive_recomp, gradient_inactive_recomp, hessian_inactive
        )
    else:
        # Step did not change, so we can just recover the
        # pre-computed prediction
        predicted_reduction = evaluate_model_criterion(
            x_inactive_cg,
            gradient_inactive_cg,
            hessian_inactive,
        )

        predicted_reduction = -predicted_reduction

    return predicted_reduction


def perform_steepest_descent_and_update_trustregion_radius(
    x_candidate,
    f_candidate,
    gradient_projected,
    hessian_inactive,
    model,
    lower_bound,
    upper_bound,
    active_bounds_info,
):
    """Perform steepest descent step and update trust-region radius."""
    x_trial = np.copy(x_candidate)
    f_min = np.copy(f_candidate)

    step_size_accepted = 0
    trustregion_radius = 100
    max_trustregion_radius = 0

    gradient_norm = np.linalg.norm(gradient_projected)

    for _ in range(5):
        x_old = np.copy(x_trial)

        # Steepest descent step
        step_size_trial = trustregion_radius / gradient_norm
        x_trial = x_trial - step_size_trial * gradient_projected
        x_trial = apply_bounds_to_x_candidate(x_trial, lower_bound, upper_bound)
        f_trial = evaluate_model_criterion(
            x_trial, model.linear_terms, model.square_terms
        )

        x_accepted = x_trial - x_old

        if f_trial < f_min:
            f_min = copy(f_trial)
            step_size_accepted = trustregion_radius / gradient_norm

        x_inactive = np.copy(x_accepted[active_bounds_info.inactive])

        square_terms = np.dot(np.dot(x_inactive, hessian_inactive), x_inactive)

        predicted_reduction = trustregion_radius * (
            gradient_norm
            - 0.5 * trustregion_radius * square_terms / (gradient_norm**2)
        )
        actual_reduction = f_candidate - f_trial

        (
            tau,
            max_trustregion_radius,
        ) = _update_tau_and_max_trustregion_radius_steepest_descent(
            trustregion_radius,
            predicted_reduction,
            actual_reduction,
            gradient_norm,
            max_trustregion_radius,
        )

        trustregion_radius = tau * trustregion_radius

    x_candidate = x_candidate - step_size_accepted * gradient_projected
    x_candidate = apply_bounds_to_x_candidate(x_candidate, lower_bound, upper_bound)

    return (
        x_candidate,
        f_min,
        trustregion_radius,
        max_trustregion_radius,
    )


def compute_newton_step(model):
    """Compute the newton step."""
    n = model.linear_terms.shape[0]
    diag_min = np.ones(n) * 1e-6
    diag_max = np.ones(n) * 1e6

    diag = np.abs(np.diag(model.square_terms))
    diag_bounded = np.median(np.stack([diag_min, diag, diag_max]), axis=0)

    newton_step = model.linear_terms / diag_bounded

    return newton_step


def update_trustregion_radius_conjugate_gradient(
    f_candidate, predicted_reduction, actual_reduction, x_norm_cg, tr_radius
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
        tr_radius = alpha1 * min(tr_radius, x_norm_cg)
    else:
        if np.isfinite(actual_reduction) is False:
            tr_radius = alpha1 * min(tr_radius, x_norm_cg)
        else:
            if abs(actual_reduction) <= max(1, abs(f_candidate) * EPSILON) and abs(
                predicted_reduction
            ) <= max(1, abs(f_candidate) * EPSILON):
                kappa = 1
            else:
                kappa = actual_reduction / predicted_reduction

            if kappa < eta1:
                tr_radius = alpha1 * min(tr_radius, x_norm_cg)
            else:
                accept_step = True

                # Update the trust-region radius only if the computed step is at the
                # trust radius boundary
                if np.allclose(x_norm_cg, tr_radius):
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


def get_information_on_active_bounds(x, lower_bound, upper_bound, model, newton_step):
    """Return the index set of active bounds."""
    ActiveBounds = namedtuple(
        "ActiveBounds", ["lower", "upper", "fixed", "all", "inactive"]
    )
    step_length = 1e-3

    x_candidate = np.copy(x)
    x_candidate = x_candidate - step_length * newton_step
    x_candidate_bounded = apply_bounds_to_x_candidate(
        x_candidate, lower_bound, upper_bound
    )

    centered_x = x - x_candidate_bounded

    active_lower = np.where((centered_x <= lower_bound) & (model.linear_terms > 0))[0]
    active_upper = np.where((centered_x >= upper_bound) & (model.linear_terms < 0))[0]
    active_fixed = np.where(lower_bound == upper_bound)[0]
    active_all = reduce(np.union1d, (active_fixed, active_lower, active_upper))
    inactive = np.setdiff1d(np.arange(x.shape[0]), active_all)

    active_bounds_info = ActiveBounds(
        lower=active_lower,
        upper=active_upper,
        fixed=active_fixed,
        all=active_all,
        inactive=inactive,
    )

    return active_bounds_info


def find_hessian_submatrix_where_bounds_inactive(model, active_bounds_info):
    """Find the submatrix of the initial hessian where bounds are inactive."""
    hessian_inactive = model.square_terms[
        active_bounds_info.inactive[:, np.newaxis], active_bounds_info.inactive
    ]

    return hessian_inactive


def check_for_convergence_conjugate_gradient(
    x_candidate,
    f_candidate,
    x_old,
    f_old,
    gradient_candidate,
    model,
    lower_bound,
    upper_bound,
    trustregion_radius,
    options,
):
    """Check if we have found a solution."""
    direction_fischer_burmeister = _get_fischer_burmeister_direction_vector(
        x_candidate, gradient_candidate, lower_bound, upper_bound
    )
    gradient_norm = np.linalg.norm(direction_fischer_burmeister)

    if trustregion_radius < options["steptol"]:
        converged = True
    elif abs(f_old - f_candidate) < options["ftol"]:
        converged = True
    elif (f_old - f_candidate) / max(abs(f_old), abs(f_candidate), 1) < options["ftol"]:
        converged = True
    elif gradient_norm < options["gtol"]:
        converged = True
    elif f_candidate != 0 and abs(gradient_norm / f_candidate) < options["gtol"]:
        converged = True
    elif gradient_norm / np.linalg.norm(model.linear_terms) < options["gtol"]:
        converged = True
    elif np.max(np.abs(x_old - x_candidate)) < options["xtol"]:
        converged = True
    else:
        converged = False

    return converged


def check_for_convergence_steepest_descent(
    x_candidate,
    f_candidate,
    gradient_candidate,
    lower_bound,
    upper_bound,
    options,
):
    """Check if we have found a solution."""
    direction_fischer_burmeister = _get_fischer_burmeister_direction_vector(
        x_candidate, gradient_candidate, lower_bound, upper_bound
    )
    gradient_norm = np.linalg.norm(direction_fischer_burmeister)

    if gradient_norm < options["gtol"]:
        converged = True
    elif f_candidate != 0 and abs(gradient_norm / f_candidate) < options["gtol"]:
        converged = True
    elif f_candidate != 0 and abs(gradient_norm / f_candidate) < options["gtol"]:
        converged = True
    else:
        converged = False

    return converged


def apply_bounds_to_x_candidate(x, lower_bound, upper_bound, bound_tol=0):
    """Apply upper and lower bounds to the candidate vector."""
    x = np.where(x <= lower_bound + bound_tol, lower_bound, x)
    x = np.where(x >= upper_bound - bound_tol, upper_bound, x)

    return x


def evaluate_model_criterion(
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


def evaluate_model_gradient(x, model):
    """Evaluate the derivative of the main model.

    Args:
        model_gradient (np.ndarray): Gradient of the main model of shape (n,).
        model_hessian (np.ndarray): Hessian of the main model of shape (n, n).

    Returns:
        (np.ndarray): derivative of the main model of shape (n,).
    """
    return model.linear_terms + np.dot(model.square_terms, x)


def _apply_bounds_to_conjugate_gradient_step(
    x,
    lower_bound,
    upper_bound,
    active_bounds_info,
    cg_step_direction=None,
):
    """Apply lower and upper bounds to the Conjugate Gradient step direction."""
    n = x.shape[0]

    if cg_step_direction is None:
        cg_step_direction = np.zeros(n)

    if active_bounds_info.lower.size > 0:
        x_active_lower = x[active_bounds_info.lower]
        lower_bound_active = lower_bound[active_bounds_info.lower]

        cg_step_direction[active_bounds_info.lower] = (
            lower_bound_active - x_active_lower
        )

    if active_bounds_info.upper.size > 0:
        x_active_upper = x[active_bounds_info.upper]
        upper_bound_active = upper_bound[active_bounds_info.upper]

        cg_step_direction[active_bounds_info.upper] = (
            upper_bound_active - x_active_upper
        )

    if active_bounds_info.fixed.size > 0:
        cg_step_direction[active_bounds_info.fixed] = 0

    return cg_step_direction


def _update_tau_and_max_trustregion_radius_steepest_descent(
    trustregion_radius, prered, actred, gradient_norm, max_radius
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
        * gradient_norm
        * trustregion_radius
        / (theta * gradient_norm * trustregion_radius + (1 - theta) * prered - actred)
    )
    tau_2 = (
        theta
        * gradient_norm
        * trustregion_radius
        / (theta * gradient_norm * trustregion_radius - (1 + theta) * prered + actred)
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


def _get_fischer_burmeister_direction_vector(x, gradient, lower_bound, upper_bound):
    """Compute the constrained direction vector via the Fischer-Burmeister function."""
    fischer_vec = np.vectorize(_get_fischer_burmeister_scalar)

    fischer_burmeister = reduce(
        fischer_vec, (upper_bound - x, -gradient, x - lower_bound)
    )
    direction = np.where(
        lower_bound == upper_bound, lower_bound - x, fischer_burmeister
    )

    return direction


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

"""Auxiliary functions for the quadratic BNTR trust-region subsolver."""
from typing import NamedTuple
from typing import Union

import numpy as np
from estimagic.optimization.subsolvers._conjugate_gradient_quadratic import (
    minimize_trust_cg,
)
from estimagic.optimization.subsolvers._steihaug_toint_quadratic import (
    minimize_trust_stcg,
)
from estimagic.optimization.subsolvers._trsbox_quadratic import minimize_trust_trsbox
from numba import njit

EPSILON = np.finfo(float).eps ** (2 / 3)


class ActiveBounds(NamedTuple):
    lower: Union[np.ndarray, None] = None
    upper: Union[np.ndarray, None] = None
    fixed: Union[np.ndarray, None] = None
    active: Union[np.ndarray, None] = None
    inactive: Union[np.ndarray, None] = None


@njit
def take_preliminary_gradient_descent_step_and_check_for_solution(
    x_candidate,
    model,
    lower_bounds,
    upper_bounds,
    maxiter_gradient_descent,
    gtol_abs,
    gtol_rel,
    gtol_scaled,
):
    """Take a preliminary gradient descent step and check if we found a solution."""

    default_radius = 100
    min_radius = 1e-10
    max_radius = 1e10
    theta = 0.25
    mu1 = 0.35
    mu2 = 0.50
    gamma1 = 0.0625
    gamma2 = 0.5
    gamma3 = 2.0
    gamma4 = 5.0

    converged = False
    convergence_reason = 0

    criterion_candidate = _evaluate_model_criterion(
        x_candidate, model.linear_terms, model.square_terms
    )

    active_bounds_info = get_information_on_active_bounds(
        x_candidate,
        model.linear_terms,
        lower_bounds,
        upper_bounds,
    )

    gradient_unprojected = _evaluate_model_gradient(x_candidate, model)
    gradient_projected = project_gradient_onto_feasible_set(
        gradient_unprojected, active_bounds_info
    )
    converged, convergence_reason = check_for_convergence(
        x_candidate=x_candidate,
        f_candidate=criterion_candidate,
        gradient_candidate=gradient_unprojected,
        model=model,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        converged=converged,
        reason=convergence_reason,
        niter=None,
        maxiter=None,
        gtol_abs=gtol_abs,
        gtol_rel=gtol_rel,
        gtol_scaled=gtol_scaled,
    )

    if converged:
        hessian_inactive = model.square_terms
        trustregion_radius = default_radius
    else:
        hessian_inactive = find_hessian_submatrix_where_bounds_inactive(
            model, active_bounds_info
        )

        (
            x_candidate_gradient_descent,
            f_min_gradient_descent,
            step_size_gradient_descent,
            trustregion_radius,
            radius_lower_bound,
        ) = perform_gradient_descent_step(
            x_candidate=x_candidate,
            f_candidate_initial=criterion_candidate,
            gradient_projected=gradient_projected,
            hessian_inactive=hessian_inactive,
            model=model,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            active_bounds_info=active_bounds_info,
            maxiter_steepest_descent=maxiter_gradient_descent,
            default_radius=default_radius,
            theta=theta,
            mu1=mu1,
            mu2=mu2,
            gamma1=gamma1,
            gamma2=gamma2,
            gamma3=gamma3,
            gamma4=gamma4,
        )

        if f_min_gradient_descent < criterion_candidate:
            criterion_candidate = f_min_gradient_descent

            x_unbounded = (
                x_candidate_gradient_descent
                - step_size_gradient_descent * gradient_projected
            )
            x_candidate = apply_bounds_to_x_candidate(
                x_unbounded, lower_bounds, upper_bounds
            )

            gradient_unprojected = _evaluate_model_gradient(x_candidate, model)
            active_bounds_info = get_information_on_active_bounds(
                x_candidate,
                gradient_unprojected,
                lower_bounds,
                upper_bounds,
            )

            gradient_projected = project_gradient_onto_feasible_set(
                gradient_unprojected, active_bounds_info
            )
            hessian_inactive = find_hessian_submatrix_where_bounds_inactive(
                model, active_bounds_info
            )

            converged, convergence_reason = check_for_convergence(
                x_candidate=x_candidate,
                f_candidate=criterion_candidate,
                gradient_candidate=gradient_projected,
                model=model,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                converged=converged,
                reason=convergence_reason,
                niter=None,
                maxiter=None,
                gtol_abs=gtol_abs,
                gtol_rel=gtol_rel,
                gtol_scaled=gtol_scaled,
            )

        if not converged:
            trustregion_radius = clip_scalar_numba(
                max(trustregion_radius, radius_lower_bound),
                min_radius,
                max_radius,
            )

    return (
        x_candidate,
        criterion_candidate,
        gradient_unprojected,
        hessian_inactive,
        trustregion_radius,
        active_bounds_info,
        converged,
        convergence_reason,
    )


def compute_conjugate_gradient_step(
    x_candidate,
    gradient_inactive,
    hessian_inactive,
    lower_bounds,
    upper_bounds,
    active_bounds_info,
    trustregion_radius,
    *,
    conjugate_gradient_method,
    gtol_abs_conjugate_gradient,
    gtol_rel_conjugate_gradient,
    options_update_radius,
):
    """Compute the bounded Conjugate Gradient trust-region step."""
    conjugate_gradient_step = np.zeros_like(x_candidate)

    if active_bounds_info.inactive.size == 0:
        # Save some computation and return an adjusted zero step
        step_inactive = apply_bounds_to_x_candidate(
            x_candidate, lower_bounds, upper_bounds
        )
        step_norm = norm_numba(step_inactive)

        conjugate_gradient_step = _apply_bounds_to_conjugate_gradient_step(
            step_inactive,
            x_candidate,
            lower_bounds,
            upper_bounds,
            active_bounds_info,
        )

    else:
        if conjugate_gradient_method == "cg":
            step_inactive = minimize_trust_cg(
                gradient_inactive,
                hessian_inactive,
                trustregion_radius,
                gtol_abs=gtol_abs_conjugate_gradient,
                gtol_rel=gtol_rel_conjugate_gradient,
            )
            step_norm = norm_numba(step_inactive)
        elif conjugate_gradient_method == "steihaug_toint":
            step_inactive = minimize_trust_stcg(
                gradient_inactive,
                hessian_inactive,
                trustregion_radius,
            )
            step_norm = norm_numba(step_inactive)
        elif conjugate_gradient_method == "trsbox":
            step_inactive = minimize_trust_trsbox(
                gradient_inactive,
                hessian_inactive,
                trustregion_radius,
                lower_bounds=lower_bounds[active_bounds_info.inactive],
                upper_bounds=upper_bounds[active_bounds_info.inactive],
            )
            step_norm = norm_numba(step_inactive)
        else:
            raise ValueError(
                "Invalid method: {conjugate_gradient_method}. "
                "Must be one of cg, steihaug_toint, trsbox."
            )

        if trustregion_radius == 0:
            if step_norm > 0:
                # Accept
                trustregion_radius = np.clip(
                    step_norm,
                    options_update_radius["min_radius"],
                    options_update_radius["max_radius"],
                )

            else:
                # Re-solve
                trustregion_radius = np.clip(
                    options_update_radius["default_radius"],
                    options_update_radius["min_radius"],
                    options_update_radius["max_radius"],
                )

                if conjugate_gradient_method == "cg":
                    step_inactive = minimize_trust_cg(
                        gradient_inactive,
                        hessian_inactive,
                        trustregion_radius,
                        gtol_abs=gtol_abs_conjugate_gradient,
                        gtol_rel=gtol_rel_conjugate_gradient,
                    )
                    step_norm = norm_numba(step_inactive)
                elif conjugate_gradient_method == "steihaug_toint":
                    step_inactive = minimize_trust_stcg(
                        gradient_inactive,
                        hessian_inactive,
                        trustregion_radius,
                    )
                    step_norm = norm_numba(step_inactive)
                elif conjugate_gradient_method == "trsbox":
                    step_inactive = minimize_trust_trsbox(
                        gradient_inactive,
                        hessian_inactive,
                        trustregion_radius,
                        lower_bounds=lower_bounds[active_bounds_info.inactive],
                        upper_bounds=upper_bounds[active_bounds_info.inactive],
                    )
                    step_norm = norm_numba(step_inactive)

                if step_norm == 0:
                    raise ValueError("Initial direction is zero.")

        conjugate_gradient_step = _apply_bounds_to_conjugate_gradient_step(
            step_inactive,
            x_candidate,
            lower_bounds,
            upper_bounds,
            active_bounds_info,
        )

    return (
        conjugate_gradient_step,
        step_inactive,
        step_norm,
    )


def compute_predicted_reduction_from_conjugate_gradient_step(
    conjugate_gradient_step,
    conjugate_gradient_step_inactive,
    gradient_unprojected,
    gradient_inactive,
    hessian_inactive,
    active_bounds_info,
):
    """Compute predicted reduction induced by the Conjugate Gradient step."""
    if active_bounds_info.active.size > 0:
        # Projection changed the step, so we have to recompute the step
        # and the predicted reduction. Leave the rust radius unchanged.
        cg_step_recomp = conjugate_gradient_step[active_bounds_info.inactive]
        gradient_inactive_recomp = gradient_unprojected[active_bounds_info.inactive]

        predicted_reduction = _evaluate_model_criterion(
            cg_step_recomp, gradient_inactive_recomp, hessian_inactive
        )
    else:
        # Step did not change, so we can just recover the
        # pre-computed prediction
        predicted_reduction = _evaluate_model_criterion(
            conjugate_gradient_step_inactive,
            gradient_inactive,
            hessian_inactive,
        )

    return -predicted_reduction


@njit
def perform_gradient_descent_step(
    x_candidate,
    f_candidate_initial,
    gradient_projected,
    hessian_inactive,
    model,
    lower_bounds,
    upper_bounds,
    active_bounds_info,
    maxiter_steepest_descent,
    default_radius,
    theta,
    mu1,
    mu2,
    gamma1,
    gamma2,
    gamma3,
    gamma4,
):
    """Perform gradient descent step and update trust-region radius."""
    f_min = f_candidate_initial
    gradient_norm = norm_numba(gradient_projected)

    trustregion_radius = default_radius
    radius_lower_bound = 0
    step_size_accepted = 0

    for _ in range(maxiter_steepest_descent):
        x_old = x_candidate

        step_size_candidate = trustregion_radius / gradient_norm
        x_candidate = x_old - step_size_candidate * gradient_projected

        x_candidate = apply_bounds_to_x_candidate(
            x_candidate, lower_bounds, upper_bounds
        )
        f_candidate = _evaluate_model_criterion(
            x_candidate, model.linear_terms, model.square_terms
        )

        x_diff = x_candidate - x_old

        if f_candidate < f_min:
            f_min = f_candidate
            step_size_accepted = step_size_candidate

        x_inactive = x_diff[active_bounds_info.inactive]
        square_terms = x_inactive.T @ hessian_inactive @ x_inactive

        predicted_reduction = trustregion_radius * (
            gradient_norm
            - 0.5 * trustregion_radius * square_terms / (gradient_norm**2)
        )
        actual_reduction = f_candidate_initial - f_candidate

        (
            trustregion_radius,
            radius_lower_bound,
        ) = _update_trustregion_radius_and_gradient_descent(
            trustregion_radius,
            radius_lower_bound,
            predicted_reduction,
            actual_reduction,
            gradient_norm,
            theta,
            mu1,
            mu2,
            gamma1,
            gamma2,
            gamma3,
            gamma4,
        )

    return (
        x_candidate,
        f_min,
        step_size_accepted,
        trustregion_radius,
        radius_lower_bound,
    )


def update_trustregion_radius_conjugate_gradient(
    f_candidate,
    predicted_reduction,
    actual_reduction,
    x_norm_cg,
    trustregion_radius,
    options,
):
    """Update the trust-region radius based on predicted and actual reduction."""
    accept_step = False

    if predicted_reduction < 0 or ~np.isfinite(predicted_reduction):
        # Reject and start over
        trustregion_radius = options["alpha1"] * min(trustregion_radius, x_norm_cg)

    else:
        if ~np.isfinite(actual_reduction):
            trustregion_radius = options["alpha1"] * min(trustregion_radius, x_norm_cg)
        else:
            if abs(actual_reduction) <= max(1, abs(f_candidate) * EPSILON) and abs(
                predicted_reduction
            ) <= max(1, abs(f_candidate) * EPSILON):
                kappa = 1
            else:
                kappa = actual_reduction / predicted_reduction

            if kappa < options["eta1"]:
                # Reject the step
                trustregion_radius = options["alpha1"] * min(
                    trustregion_radius, x_norm_cg
                )
            else:
                accept_step = True

                # Update the trust-region radius only if the computed step is at the
                # trust-radius boundary
                if x_norm_cg == trustregion_radius:
                    if kappa < options["eta2"]:
                        # Marginal bad step
                        trustregion_radius = options["alpha2"] * trustregion_radius
                    elif kappa < options["eta3"]:
                        # Reasonable step
                        trustregion_radius = options["alpha3"] * trustregion_radius
                    elif kappa < options["eta4"]:
                        trustregion_radius = options["alpha4"] * trustregion_radius
                    else:
                        # Very good step
                        trustregion_radius = options["alpha5"] * trustregion_radius

    trustregion_radius = np.clip(
        trustregion_radius, options["min_radius"], options["max_radius"]
    )

    return trustregion_radius, accept_step


@njit
def get_information_on_active_bounds(
    x,
    gradient_unprojected,
    lower_bounds,
    upper_bounds,
):
    """Return the index set of active bounds."""
    active_all = []
    active_upper = []
    active_lower = []
    active_fixed = []
    inactive = []
    for i in range(len(x)):
        if (x[i] <= lower_bounds[i]) & (gradient_unprojected[i] > 0):
            if i not in active_all:
                active_all.append(i)
            if i not in active_lower:
                active_lower.append(i)
        elif (x[i] >= upper_bounds[i]) & (gradient_unprojected[i] < 0):
            if i not in active_all:
                active_all.append(i)
            if i not in active_upper:
                active_upper.append(i)
        elif lower_bounds[i] == upper_bounds[i]:
            if i not in active_fixed:
                active_fixed.append(i)
            if i not in active_all:
                active_all.append(i)
        else:
            if i not in inactive:
                inactive.append(i)
    active_all = np.array(active_all)
    active_lower = np.array(active_lower)
    active_upper = np.array(active_upper)
    inactive = np.array(inactive)
    active_fixed = np.array(active_fixed)
    active_bounds_info = ActiveBounds(
        lower=active_lower,
        upper=active_upper,
        fixed=active_fixed,
        active=active_all,
        inactive=inactive,
    )

    return active_bounds_info


@njit
def find_hessian_submatrix_where_bounds_inactive(model, active_bounds_info):
    """Find the submatrix of the initial hessian where bounds are inactive."""
    inactive_bounds = active_bounds_info.inactive
    initial_hessian = model.square_terms
    hessian_inactive = initial_hessian[:, inactive_bounds]
    hessian_inactive = hessian_inactive[inactive_bounds, :]
    return hessian_inactive


@njit
def check_for_convergence(
    x_candidate,
    f_candidate,
    gradient_candidate,
    model,
    lower_bounds,
    upper_bounds,
    converged,
    reason,
    niter,
    maxiter,
    gtol_abs,
    gtol_rel,
    gtol_scaled,
):
    """Check if we have found a solution."""
    direction_fischer_burmeister = _get_fischer_burmeister_direction_vector(
        x_candidate, gradient_candidate, lower_bounds, upper_bounds
    )
    gradient_norm = norm_numba(direction_fischer_burmeister)
    gradient_norm_initial = norm_numba(model.linear_terms)

    if gradient_norm < gtol_abs:
        converged = True
        reason = 1
    elif f_candidate != 0 and abs(gradient_norm / f_candidate) < gtol_rel:
        converged = True
        reason = 2
    elif (
        gradient_norm_initial != 0
        and gradient_norm / gradient_norm_initial < gtol_scaled
    ):
        converged = True
        reason = 3
    elif gradient_norm_initial != 0 and gradient_norm == 0 and gtol_scaled == 0:
        converged = True
        reason = 4
    elif f_candidate <= -np.inf:
        converged = True
        reason = 5
    elif niter is not None and niter == maxiter:
        reason = 6

    return converged, reason


@njit
def apply_bounds_to_x_candidate(x, lower_bounds, upper_bounds, bound_tol=0):
    """Apply upper and lower bounds to the candidate vector."""
    for i in range(len(x)):
        if x[i] <= lower_bounds[i] + bound_tol:
            x[i] = lower_bounds[i]
        elif x[i] >= upper_bounds[i] - bound_tol:
            x[i] = upper_bounds[i]
    return x


@njit
def project_gradient_onto_feasible_set(gradient_unprojected, active_bounds_info):
    """Project gradient onto feasible set, where search directions unconstrained."""
    gradient_projected = np.zeros(len(gradient_unprojected))
    gradient_projected[active_bounds_info.inactive] = gradient_unprojected[
        active_bounds_info.inactive
    ]

    return gradient_projected


def _apply_bounds_to_conjugate_gradient_step(
    step_inactive,
    x_candidate,
    lower_bounds,
    upper_bounds,
    active_bounds_info,
):
    """Apply lower and upper bounds to the Conjugate Gradient step."""
    cg_step = np.zeros_like(x_candidate)
    cg_step[active_bounds_info.inactive] = step_inactive

    if active_bounds_info.lower.size > 0:
        x_active_lower = x_candidate[active_bounds_info.lower]
        lower_bound_active = lower_bounds[active_bounds_info.lower]

        cg_step[active_bounds_info.lower] = lower_bound_active - x_active_lower

    if active_bounds_info.upper.size > 0:
        x_active_upper = x_candidate[active_bounds_info.upper]
        upper_bound_active = upper_bounds[active_bounds_info.upper]

        cg_step[active_bounds_info.upper] = upper_bound_active - x_active_upper

    if active_bounds_info.fixed.size > 0:
        cg_step[active_bounds_info.fixed] = 0

    return cg_step


@njit
def _update_trustregion_radius_and_gradient_descent(
    trustregion_radius,
    radius_lower_bound,
    predicted_reduction,
    actual_reduction,
    gradient_norm,
    theta,
    mu1,
    mu2,
    gamma1,
    gamma2,
    gamma3,
    gamma4,
):
    """Update the trust-region radius and its upper bound."""
    if np.abs(actual_reduction) <= EPSILON and np.abs(predicted_reduction) <= EPSILON:
        kappa = 1
    else:
        kappa = actual_reduction / predicted_reduction

    tau_1 = (
        theta
        * gradient_norm
        * trustregion_radius
        / (
            theta * gradient_norm * trustregion_radius
            + (1 - theta) * predicted_reduction
            - actual_reduction
        )
    )
    tau_2 = (
        theta
        * gradient_norm
        * trustregion_radius
        / (
            theta * gradient_norm * trustregion_radius
            - (1 + theta) * predicted_reduction
            + actual_reduction
        )
    )

    tau_min = min(tau_1, tau_2)
    tau_max = max(tau_1, tau_2)

    if np.abs(kappa - 1) <= mu1:
        # Great agreement
        radius_lower_bound = max(radius_lower_bound, trustregion_radius)

        if tau_max < 1:
            tau = gamma3
        elif tau_max > gamma4:
            tau = gamma4
        else:
            tau = tau_max

    elif np.abs(kappa - 1) <= mu2:
        # Good agreement
        radius_lower_bound = max(radius_lower_bound, trustregion_radius)

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

    trustregion_radius = trustregion_radius * tau

    return trustregion_radius, radius_lower_bound


@njit
def _get_fischer_burmeister_direction_vector(x, gradient, lower_bounds, upper_bounds):
    """Compute the constrained direction vector via the Fischer-Burmeister function."""

    direction = []
    for x_, g_, l_, u_ in zip(x, gradient, lower_bounds, upper_bounds):
        fischer_scalar = _get_fischer_burmeister_scalar(u_ - x_, -g_)
        fischer_scalar = _get_fischer_burmeister_scalar(fischer_scalar, x_ - l_)

        if l_ == u_:
            direction.append(l_ - x_)
        else:
            direction.append(fischer_scalar)
    return np.array(direction)


@njit
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


@njit
def _evaluate_model_criterion(
    x,
    gradient,
    hessian,
):
    """Evaluate the criterion function value of the main model.

    Args:
        x (np.ndarray): Parameter vector of shape (n,).
        gradient (np.ndarray): Gradient of shape (n,) for which the main model
            shall be evaluated.
        hessian (np.ndarray): Hessian of shape (n, n) for which the main model
            shall be evaulated.

    Returns:
        float: Criterion value of the main model.
    """
    return gradient.T @ x + 0.5 * x.T @ hessian @ x


@njit
def _evaluate_model_gradient(x, model):
    """Evaluate the derivative of the main model.

    Args:
       main_model (NamedTuple): Named tuple containing the parameters of the
            main model, i.e.:
            - ``linear_terms``, a np.ndarray of shape (n,) and
            - ``square_terms``, a np.ndarray of shape (n,n).

    Returns:
        np.ndarray: Derivative of the main model of shape (n,).
    """
    return model.linear_terms + model.square_terms @ x


@njit
def clip_array_numba(x, a, b):
    """Numba-compatible clipping function for arrays."""
    for i in range(len(x)):
        if x[i] <= a[i]:
            x[i] = a[i]
        elif x[i] >= b[i]:
            x[i] = b[i]
    return x


@njit
def clip_scalar_numba(x, a, b):
    """Numba-compatible clipping function for scalars."""
    clipped = max(x, a)
    clipped = min(clipped, b)
    return clipped


@njit
def norm_numba(x):
    """Calculate Ecuildean norm of vector."""
    x = x.reshape(
        x.size,
    )
    sum_squares = 0
    for i in range(len(x)):
        sum_squares += x[i] ** 2
    norm = np.sqrt(sum_squares)
    return norm

"""Implement the POUNDERS algorithm"""
import warnings

import numpy as np
from estimagic.batch_evaluators import process_batch_evaluator
from estimagic.config import DEFAULT_N_CORES
from estimagic.decorators import mark_minimizer
from estimagic.optimization.history import LeastSquaresHistory
from estimagic.optimization.pounders_auxiliary import (
    add_geomtery_points_to_make_main_model_fully_linear,
)
from estimagic.optimization.pounders_auxiliary import create_initial_residual_model
from estimagic.optimization.pounders_auxiliary import create_main_from_residual_model
from estimagic.optimization.pounders_auxiliary import find_affine_points
from estimagic.optimization.pounders_auxiliary import get_coefficients_residual_model
from estimagic.optimization.pounders_auxiliary import (
    get_interpolation_matrices_residual_model,
)
from estimagic.optimization.pounders_auxiliary import (
    get_last_model_indices_and_check_for_repeated_model,
)
from estimagic.optimization.pounders_auxiliary import interpolate_residual_model
from estimagic.optimization.pounders_auxiliary import solve_subproblem
from estimagic.optimization.pounders_auxiliary import (
    update_main_model_with_new_accepted_x,
)
from estimagic.optimization.pounders_auxiliary import (
    update_model_indices_residual_model,
)
from estimagic.optimization.pounders_auxiliary import update_residual_model
from estimagic.optimization.pounders_auxiliary import (
    update_residual_model_with_new_accepted_x,
)
from estimagic.optimization.pounders_auxiliary import update_trustregion_radius


@mark_minimizer(
    name="pounders",
    primary_criterion_entry="root_contributions",
    needs_scaling=True,
    is_available=True,
)
def pounders(
    criterion,
    x,
    lower_bounds,
    upper_bounds,
    convergence_absolute_gradient_tolerance=1e-8,
    convergence_relative_gradient_tolerance=1e-8,
    convergence_scaled_gradient_tolerance=False,
    max_interpolation_points=None,
    stopping_max_iterations=2_000,
    trustregion_initial_radius=0.1,
    trustregion_minimal_radius=1e-6,
    trustregion_maximal_radius=1e6,
    trustregion_shrinking_factor_not_successful=0.5,
    trustregion_expansion_factor_successful=2,
    theta1=1e-5,
    theta2=1e-4,
    trustregion_threshold_acceptance=0,
    trustregion_threshold_successful=0.1,
    c1=None,
    c2=10,
    trustregion_subproblem_solver="bntr",
    trustregion_subsolver_options=None,
    batch_evaluator="joblib",
    n_cores=DEFAULT_N_CORES,
):
    """Find the local minimum to a non-linear least-squares problem using POUNDERS.
    For details, see :ref:`_own_algorithms`.

    """
    batch_evaluator = process_batch_evaluator(batch_evaluator)

    if max_interpolation_points is None:
        max_interpolation_points = 2 * len(x) + 1

    if c1 is None:
        c1 = np.sqrt(x.shape[0])

    if trustregion_subsolver_options is None:
        trustregion_subsolver_options = {}

    default_options = {
        "conjugate_gradient_method": "trsbox",
        "maxiter": 50,
        "maxiter_gradient_descent": 5,
        "gtol_abs": 1e-8,
        "gtol_rel": 1e-8,
        "gtol_scaled": 0,
        "gtol_abs_cg": 1e-8,
        "gtol_rel_cg": 1e-6,
        "k_easy": 0.1,
        "k_hard": 0.2,
    }
    trustregion_subsolver_options = {
        **default_options,
        **trustregion_subsolver_options,
    }

    result = internal_solve_pounders(
        criterion=criterion,
        x0=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        gtol_abs=convergence_absolute_gradient_tolerance,
        gtol_rel=convergence_relative_gradient_tolerance,
        gtol_scaled=convergence_scaled_gradient_tolerance,
        maxinterp=max_interpolation_points,
        maxiter=stopping_max_iterations,
        delta=trustregion_initial_radius,
        delta_min=trustregion_minimal_radius,
        delta_max=trustregion_maximal_radius,
        gamma0=trustregion_shrinking_factor_not_successful,
        gamma1=trustregion_expansion_factor_successful,
        theta1=theta1,
        theta2=theta2,
        eta0=trustregion_threshold_acceptance,
        eta1=trustregion_threshold_successful,
        c1=c1,
        c2=c2,
        solver_sub=trustregion_subproblem_solver,
        conjugate_gradient_method_sub=trustregion_subsolver_options[
            "conjugate_gradient_method"
        ],
        maxiter_sub=trustregion_subsolver_options["maxiter"],
        maxiter_gradient_descent_sub=trustregion_subsolver_options[
            "maxiter_gradient_descent"
        ],
        gtol_abs_sub=trustregion_subsolver_options["gtol_abs"],
        gtol_rel_sub=trustregion_subsolver_options["gtol_rel"],
        gtol_scaled_sub=trustregion_subsolver_options["gtol_scaled"],
        gtol_abs_conjugate_gradient_sub=trustregion_subsolver_options["gtol_abs_cg"],
        gtol_rel_conjugate_gradient_sub=trustregion_subsolver_options["gtol_rel_cg"],
        k_easy_sub=trustregion_subsolver_options["k_easy"],
        k_hard_sub=trustregion_subsolver_options["k_hard"],
        batch_evaluator=batch_evaluator,
        n_cores=n_cores,
    )

    return result


def internal_solve_pounders(
    criterion,
    x0,
    lower_bounds,
    upper_bounds,
    gtol_abs,
    gtol_rel,
    gtol_scaled,
    maxinterp,
    maxiter,
    delta,
    delta_min,
    delta_max,
    gamma0,
    gamma1,
    theta1,
    theta2,
    eta0,
    eta1,
    c1,
    c2,
    solver_sub,
    conjugate_gradient_method_sub,
    maxiter_sub,
    maxiter_gradient_descent_sub,
    gtol_abs_sub,
    gtol_rel_sub,
    gtol_scaled_sub,
    gtol_abs_conjugate_gradient_sub,
    gtol_rel_conjugate_gradient_sub,
    k_easy_sub,
    k_hard_sub,
    batch_evaluator,
    n_cores,
):
    """Find the local minimum to a non-linear least-squares problem using POUNDERS.

    Args:
        criterion_and_derivative (callable): Function that returns criterion
            and derivative as a tuple.
        x0 (np.ndarray): Initial guess for the parameter vector (starting points).
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds
            for the parameter vector x.
        gtol_abs (float): Convergence tolerance for the absolute gradient norm.
            Stop if norm of the gradient is less than this.
        gtol_rel (float): Convergence tolerance for the relative gradient norm.
            Stop if norm of the gradient relative to the criterion value is less
            than this.
        gtol_scaled (float): Convergence tolerance for the scaled gradient norm.
            Stop if norm of the gradient divided by norm of the gradient at the
            initial parameters is less than this.
        maxinterp (int): Maximum number of interpolation points.
            Default is `2 * n + 1`, where `n` is the length of the parameter vector.
        maxiter (int): Maximum number of iterations. If reached, terminate.
        delta (float): Delta, initial trust-region radius.
        delta_min (float): Minimal trust-region radius.
        delta_max (float): Maximal trust-region radius.
        gamma0 (float): Shrinking factor of the trust-region radius in case the
            solution vector of the suproblem is not accepted, but the model is fully
            linar (i.e. "valid").
        gamma1 (float): Expansion factor of the trust-region radius in case the
            solution vector of the suproblem is accepted.
        theta1 (float): Threshold for adding the current candidate vector
            to the model. Function argument to find_affine_points().
        theta2 (float): Threshold for adding the current candidate vector
            to the model. Argument to get_interpolation_matrices_residual_model().
        eta0 (float): Threshold for accepting the solution vector of the trust-region
            subproblem as the best candidate.
        eta1 (float): Threshold for successfully accepting the solution vector of the
            trust-region subproblem as the best candidate.
        c1 (float): Treshold for accepting the norm of our current x candidate.
            Equal to sqrt(n) by default. Argument to find_affine_points() in case
            the input array *model_improving_points* is zero.
        c2 (int)): Treshold for accepting the norm of our current candidate vector.
            Equal to 10 by default. Argument to find_affine_points() in case
            the input array *model_improving_points* is not zero.
        solver_sub (str): Solver to use for the trust-region subproblem.
            Two internal solvers are supported:
            - "bntr": Bounded Newton Trust-Region (default, supports bound constraints)
            - "gqtpar": (does not support bound constraints)
        conjugate_gradient_method_sub (str): Method for computing the conjugate
            gradient step ("bntr").
            Available conjugate gradient methods are:
                - "cg"
                - "steihaug_toint"
                - "trsbox" (default)
        maxiter_sub (int): Maximum number of iterations in the trust-region subproblem.
        maxiter_gradient_descent_sub (int): Maximum number of gradient descent
            iterations to perform ("bntr").
        gtol_abs_sub (float): Convergence tolerance for the absolute gradient norm
            in the trust-region subproblem ("bntr").
        gtol_rel_sub (float): Convergence tolerance for the relative gradient norm
            in the trust-region subproblem ("bntr").
        gtol_scaled_sub (float): Convergence tolerance for the scaled gradient norm
            in the trust-region subproblem ("bntr").
        gtol_abs_conjugate_gradient_sub (float): Convergence tolerance for the
            absolute gradient norm in the conjugate gradient step of the trust-region
            subproblem if "cg" is used as ``conjugate_gradient_method_sub`` ("bntr").
        gtol_rel_conjugate_gradient_sub (float): Convergence tolerance for the
            relative gradient norm in the conjugate gradient step of the trust-region
            subproblem if "cg" is used as ``conjugate_gradient_method_sub`` ("bntr").
        k_easy_sub (float): topping criterion for the "easy" case in the trust-region
            subproblem ("gqtpar").
        k_hard_sub (float): Stopping criterion for the "hard" case in the trust-region
            subproblem ("gqtpar").
        batch_evaluator (str or callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or callable with the same interface
            as the estimagic batch_evaluators.
        n_cores (int): Number of processes used to parallelize the function
            evaluations. Default is 1.

    Returns:
        (dict) Result dictionary containing:
        - solution_x (np.ndarray): Solution vector of shape (n,).
        - solution_criterion (np.ndarray): Values of the criterion function at the
            solution vector. Shape (n_obs,).
        - history_x (np.ndarray): Entire history of x. Shape (history.get_n_fun(), n).
        - history_criterion (np.ndarray): Entire history of the criterion function
            evaluations. Shape (history.get_n_fun(), n_obs)
        - n_iterations (int): Number of iterations the algorithm ran before finding a
            solution vector or reaching maxiter.
        - success (bool): Boolean indicating whether a solution has been found
            before reaching maxiter.
    """
    history = LeastSquaresHistory()

    n = len(x0)
    model_indices = np.zeros(maxinterp, dtype=int)

    n_last_modelpoints = 0

    if lower_bounds is not None and upper_bounds is not None:
        if np.max(x0 + delta - upper_bounds) > 1e-10:
            raise ValueError("Starting points + delta > upper bounds.")

    xs = [x0]
    for i in range(n):
        x1 = x0.copy()
        x1[i] += delta
        xs.append(x1)

    residuals = batch_evaluator(criterion, arguments=xs, n_cores=n_cores)

    history.add_entries(xs, residuals)
    accepted_index = history.get_best_index()

    residual_model = create_initial_residual_model(
        history=history, accepted_index=accepted_index, delta=delta
    )
    main_model = create_main_from_residual_model(
        residual_model=residual_model, multiply_square_terms_with_residuals=False
    )

    x_accepted = history.get_best_x()
    gradient_norm_initial = np.linalg.norm(main_model.linear_terms)
    gradient_norm_initial *= delta

    valid = True
    n_modelpoints = n + 1
    last_model_indices = np.zeros(maxinterp, dtype=int)

    converged = False
    convergence_reason = "Continue iterating."

    for niter in range(maxiter + 1):
        result_sub = solve_subproblem(
            x_accepted=x_accepted,
            main_model=main_model,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            delta=delta,
            solver=solver_sub,
            conjugate_gradient_method=conjugate_gradient_method_sub,
            maxiter=maxiter_sub,
            maxiter_gradient_descent=maxiter_gradient_descent_sub,
            gtol_abs=gtol_abs_sub,
            gtol_rel=gtol_rel_sub,
            gtol_scaled=gtol_scaled_sub,
            gtol_abs_conjugate_gradient=gtol_abs_conjugate_gradient_sub,
            gtol_rel_conjugate_gradient=gtol_rel_conjugate_gradient_sub,
            k_easy=k_easy_sub,
            k_hard=k_hard_sub,
        )

        x_candidate = x_accepted + result_sub["x"] * delta
        residuals_candidate = criterion(x_candidate)
        history.add_entries(x_candidate, residuals_candidate)

        predicted_reduction = history.get_critvals(
            accepted_index
        ) - history.get_critvals(-1)
        actual_reduction = -result_sub["criterion"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rho = np.divide(predicted_reduction, actual_reduction)

        if (rho >= eta1) or (rho > eta0 and valid is True):
            residual_model = residual_model._replace(
                intercepts=history.get_residuals(index=accepted_index)
            )
            center_info = {"x": history.get_best_x(), "radius": delta}
            x_candidate = history.get_centered_xs(center_info, index=-1)

            residual_model = update_residual_model_with_new_accepted_x(
                residual_model=residual_model, x_candidate=x_candidate
            )
            main_model = update_main_model_with_new_accepted_x(
                main_model=main_model, x_candidate=x_candidate
            )
            x_accepted = history.get_best_x()
            accepted_index = history.get_best_index()

        critval_accepted = history.get_critvals(index=accepted_index)

        # The model is deemend "not valid" if it has less than n model points.
        # Otherwise, if the model has n points, it is considered "valid" or
        # "fully linear".
        # Note: valid is True in the first iteration
        if not valid:
            (
                model_improving_points,
                model_indices,
                n_modelpoints,
                project_x_onto_null,
            ) = find_affine_points(
                history=history,
                x_accepted=x_accepted,
                model_improving_points=np.zeros((n, n)),
                project_x_onto_null=False,
                delta=delta,
                theta1=theta1,
                c=c1,
                model_indices=model_indices,
                n_modelpoints=0,
            )

            if n_modelpoints < n:
                (
                    history,
                    model_indices,
                ) = add_geomtery_points_to_make_main_model_fully_linear(
                    history=history,
                    main_model=main_model,
                    model_improving_points=model_improving_points,
                    model_indices=model_indices,
                    x_accepted=x_accepted,
                    n_modelpoints=n_modelpoints,
                    delta=delta,
                    criterion=criterion,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    batch_evaluator=batch_evaluator,
                    n_cores=n_cores,
                )
                n_modelpoints = n

        delta_old = delta
        delta = update_trustregion_radius(
            result_subproblem=result_sub,
            rho=rho,
            model_is_valid=valid,
            delta=delta,
            delta_min=delta_min,
            delta_max=delta_max,
            eta1=eta1,
            gamma0=gamma0,
            gamma1=gamma1,
        )

        (
            model_improving_points,
            model_indices,
            n_modelpoints,
            project_x_onto_null,
        ) = find_affine_points(
            history=history,
            x_accepted=x_accepted,
            model_improving_points=np.zeros((n, n)),
            project_x_onto_null=False,
            delta=delta,
            theta1=theta1,
            c=c1,
            model_indices=model_indices,
            n_modelpoints=0,
        )

        if n_modelpoints == n:
            valid = True
        else:
            valid = False
            (
                model_improving_points,
                model_indices,
                n_modelpoints,
                project_x_onto_null,
            ) = find_affine_points(
                history=history,
                x_accepted=x_accepted,
                model_improving_points=model_improving_points,
                project_x_onto_null=project_x_onto_null,
                delta=delta,
                theta1=theta1,
                c=c2,
                model_indices=model_indices,
                n_modelpoints=n_modelpoints,
            )

            if n_modelpoints < n:
                (
                    history,
                    model_indices,
                ) = add_geomtery_points_to_make_main_model_fully_linear(
                    history=history,
                    main_model=main_model,
                    model_improving_points=model_improving_points,
                    model_indices=model_indices,
                    x_accepted=x_accepted,
                    n_modelpoints=n_modelpoints,
                    delta=delta,
                    criterion=criterion,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                    batch_evaluator=batch_evaluator,
                    n_cores=n_cores,
                )

        model_indices, n_modelpoints = update_model_indices_residual_model(
            model_indices, accepted_index, n_modelpoints
        )

        (
            x_sample_monomial_basis,
            monomial_basis,
            basis_null_space,
            lower_triangular,
            n_modelpoints,
        ) = get_interpolation_matrices_residual_model(
            history=history,
            x_accepted=x_accepted,
            model_indices=model_indices,
            delta=delta,
            c2=c2,
            theta2=theta2,
            n_maxinterp=maxinterp,
            n_modelpoints=n_modelpoints,
        )

        center_info = {"x": x_accepted, "radius": delta_old}
        interpolation_set = history.get_centered_xs(
            center_info, index=model_indices[:n_modelpoints]
        )

        residual_model_interpolated = interpolate_residual_model(
            history=history,
            interpolation_set=interpolation_set,
            residual_model=residual_model,
            model_indices=model_indices,
            n_modelpoints=n_modelpoints,
            n_maxinterp=maxinterp,
        )
        coefficients_residual_model = get_coefficients_residual_model(
            x_sample_monomial_basis=x_sample_monomial_basis,
            monomial_basis=monomial_basis,
            basis_null_space=basis_null_space,
            lower_triangular=lower_triangular,
            residual_model_interpolated=residual_model_interpolated,
            n_modelpoints=n_modelpoints,
        )

        residual_model = residual_model._replace(
            intercepts=history.get_residuals(index=accepted_index)
        )
        residual_model = update_residual_model(
            residual_model=residual_model,
            coefficients_to_add=coefficients_residual_model,
            delta=delta,
            delta_old=delta_old,
        )
        main_model = create_main_from_residual_model(residual_model)

        gradient_norm = np.linalg.norm(main_model.linear_terms)
        gradient_norm *= delta

        (
            last_model_indices,
            n_last_modelpoints,
            same_model_used,
        ) = get_last_model_indices_and_check_for_repeated_model(
            model_indices=model_indices,
            last_model_indices=last_model_indices,
            n_modelpoints=n_modelpoints,
            n_last_modelpoints=n_last_modelpoints,
        )

        converged, convergence_reason = _check_for_convergence(
            gradient_norm=gradient_norm,
            gradient_norm_initial=gradient_norm_initial,
            critval=critval_accepted,
            delta=delta,
            delta_old=delta_old,
            same_model_used=same_model_used,
            converged=converged,
            reason=convergence_reason,
            niter=niter,
            gtol_abs=gtol_abs,
            gtol_rel=gtol_rel,
            gtol_scaled=gtol_scaled,
            maxiter=maxiter,
        )

        if converged:
            break

    result_dict = {
        "solution_x": history.get_xs(index=accepted_index),
        "solution_criterion": history.get_best_residuals(),
        "history_x": history.get_xs(),
        "history_criterion": history.get_residuals(),
        "n_iterations": niter,
        "success": converged,
        "message": convergence_reason,
    }

    return result_dict


def _check_for_convergence(
    gradient_norm,
    gradient_norm_initial,
    critval,
    delta,
    delta_old,
    same_model_used,
    converged,
    reason,
    niter,
    *,
    gtol_abs,
    gtol_rel,
    gtol_scaled,
    maxiter,
):
    """Check for convergence."""
    if same_model_used and delta == delta_old:
        converged = True
        reason = "Identical model used in successive iterations."
    elif gradient_norm < gtol_abs:
        converged = True
        reason = "Norm of the gradient is less than absolute_gradient_tolerance."
    elif critval != 0 and abs(gradient_norm / critval) < gtol_rel:
        converged = True
        reason = (
            "Norm of the gradient relative to the criterion value is less than "
            "relative_gradient_tolerance."
        )
    elif (
        gradient_norm_initial != 0
        and gradient_norm / gradient_norm_initial < gtol_scaled
    ):
        converged = True
        reason = (
            "Norm of the gradient divided by norm of the gradient at the "
            "initial parameters is less than scaled_gradient_tolerance."
        )
    elif gradient_norm_initial != 0 and gradient_norm == 0 and gtol_scaled == 0:
        converged = True
        reason = (
            "Norm of the gradient divided by norm of the gradient at the "
            "initial parameters is less than scaled_gradient_tolerance."
        )
    elif critval <= -np.inf:
        converged = True
        reason = "Criterion value is negative infinity."
    elif niter == maxiter:
        reason = "Maximum number of iterations reached."

    return converged, reason

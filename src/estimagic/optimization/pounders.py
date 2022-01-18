"""Implement the pounders algorithm"""
from functools import partial

import estimagic.batch_evaluators as be
import numpy as np
from estimagic.config import DEFAULT_N_CORES
from estimagic.optimization.history import LeastSquaresHistory
from estimagic.optimization.pounders_auxiliary import (
    add_points_to_make_main_model_fully_linear,
)
from estimagic.optimization.pounders_auxiliary import find_affine_points
from estimagic.optimization.pounders_auxiliary import get_coefficients_residual_model
from estimagic.optimization.pounders_auxiliary import (
    get_interpolation_matrices_residual_model,
)
from estimagic.optimization.pounders_auxiliary import interpolate_f
from estimagic.optimization.pounders_auxiliary import solve_subproblem
from estimagic.optimization.pounders_auxiliary import update_initial_residual_model
from estimagic.optimization.pounders_auxiliary import update_main_from_residual_model
from estimagic.optimization.pounders_auxiliary import (
    update_main_model_with_new_accepted_x,
)
from estimagic.optimization.pounders_auxiliary import update_residual_model
from estimagic.optimization.pounders_auxiliary import (
    update_residual_model_with_new_accepted_x,
)


def pounders(
    criterion_and_derivative,
    x,
    lower_bounds,
    upper_bounds,
    convergence_absolute_gradient_tolerance=1e-4,
    stopping_max_iterations=200,
    trustregion_initial_radius=0.1,
    trustregion_minimal_radius=1e-6,
    trustregion_maximal_radius=1e6,
    trustregion_shrinking_factor_not_successful=0.5,
    trustregion_expansion_factor_successful=2,
    theta1=1e-5,
    theta2=1e-4,
    trustregion_threshold_successful=0,
    trustregion_threshold_very_successful=0.1,
    c1=None,
    c2=10,
    trustregion_subproblem_solver="trust-constr",
    trustregion_subproblem_options=None,
    batch_evaluator="joblib",
    n_cores=DEFAULT_N_CORES,
):
    """Find the local minimum to a non-linear least-squares problem using POUNDERS.

    For details see :ref:`list_of_scipy_algorithms`.

    """
    if isinstance(batch_evaluator, str):
        batch_evaluator = getattr(be, f"{batch_evaluator}_batch_evaluator")

    algorithm_info = {
        "primary_criterion_entry": "root_contributions",
        "parallelizes": False,
        "needs_scaling": True,
        "name": "pounders",
    }
    criterion = partial(
        criterion_and_derivative, algorithm_info=algorithm_info, task="criterion"
    )

    if c1 is None:
        c1 = np.sqrt(x.shape[0])

    if trustregion_subproblem_options is None:
        trustregion_subproblem_options = {}

    default_options = {"ftol": 1e-6, "xtol": 1e-6, "gtol": 1e-6}
    trustregion_subproblem_options = {
        **default_options,
        **trustregion_subproblem_options,
    }

    result_sub = internal_solve_pounders(
        criterion=criterion,
        x0=x,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        gtol=convergence_absolute_gradient_tolerance,
        maxiter=stopping_max_iterations,
        delta=trustregion_initial_radius,
        delta_min=trustregion_minimal_radius,
        delta_max=trustregion_maximal_radius,
        gamma0=trustregion_shrinking_factor_not_successful,
        gamma1=trustregion_expansion_factor_successful,
        theta1=theta1,
        theta2=theta2,
        eta0=trustregion_threshold_successful,
        eta1=trustregion_threshold_very_successful,
        c1=c1,
        c2=c2,
        solver_sub=trustregion_subproblem_solver,
        ftol_sub=trustregion_subproblem_options["ftol"],
        xtol_sub=trustregion_subproblem_options["xtol"],
        gtol_sub=trustregion_subproblem_options["gtol"],
        batch_evaluator=batch_evaluator,
        n_cores=n_cores,
    )

    return result_sub


def internal_solve_pounders(
    criterion,
    x0,
    lower_bounds,
    upper_bounds,
    gtol,
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
    ftol_sub,
    xtol_sub,
    gtol_sub,
    batch_evaluator,
    n_cores,
):
    """Minimize the criterion function using POUNDERS.

    Args:
        criterion_and_derivative (callable): Function that returns criterion
            and derivative as a tuple.
        x0 (np.ndarray): Initial guess of the parameter vector. Starting points.
        lower_bounds (np.ndarray): Lower bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.
        gtol (float): Convergence criterion of the absolute gradient norm.
            Default is 1e-4.
        maxiter (int): Maximum number of iterations. If reached, terminate.
        delta (float): Delta, initial trust-region radius.
        delta_min (float): Minimal trust-region radius.
        delta_max (float): Maximal trust-region radius.
        gamma0 (float): Shrinking factor of the trust-region radius in case the
            solution vector of the suproblem is not accepted, but the model is fully
            linar (i.e. "valid").
        gamma1 (float): Expansion factor of the trust-region radius in case the
            solution vector of the suproblem is accepted.
        theta1 (float): Threshold for adding the current x candidate
            to the model. Function argument to find_affine_points().
        theta2 (float): Threshold for adding the current x candidate
            to the model. Argument to get_interpolation_matrices_residual_model().
        eta0 (float): First threshold for accepting the solution vector of the
            subproblem as the best x candidate.
        eta1 (float): Second threshold for accepting the solution vector of the
            subproblem as the best x candidate.
        c1 (float): Treshold for accepting the norm of our current x candidate.
            Equal to sqrt(n) by default. Argument to find_affine_points() in case
            the input array *model_improving_points* is zero.
        c2 (int)): Treshold for accepting the norm of our current x candidate.
            Equal to 10 by default. Argument to find_affine_points() in case
            the input array *model_improving_points* is not zero.
        solver_sub (str): Bound-constraint minimizer for the subproblem.
            Currently, three solvers from the scipy library are supported.
            - "trust-constr" (default)
            - "L-BFGS-B"
            - "SLSQP"
        ftol_sub (float): Tolerance for f, the criterion function value.
            Stopping criterion for the subproblem.
        xtol_sub (float): Tolerance for solution vector x.
            Stopping criterion for the subproblem.
        gtol_sub (float): Tolerance for the absolute gradient norm.
            Stopping criterion for the subproblem.
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
        - message (str): Message to the user. Currently it says: "Under development."
    """
    history = LeastSquaresHistory()

    n = x0.shape[0]
    n_maxinterp = 2 * n + 1
    model_indices = np.zeros(n_maxinterp, dtype=int)

    last_n_modelpoints = 0
    niter = 0

    if lower_bounds is not None and upper_bounds is not None:
        if np.max(x0 + delta - upper_bounds) > 1e-10:
            raise ValueError("Starting points + delta > upper bounds.")

    xs = [x0]
    for i in range(n):
        x1 = np.copy(x0)
        x1[i] += delta
        xs.append(x1)

    residuals = batch_evaluator(criterion, arguments=xs, n_cores=n_cores)

    history.add_entries(xs, residuals)
    accepted_index = history.get_best_index()

    # Center around new trust-region and normalize to [-1, 1]
    indices_not_min = [i for i in range(n + 1) if i != accepted_index]

    center_info = {
        "x": history.get_best_x(),
        "residuals": history.get_best_residuals(),
        "radius": delta,
    }
    x_candidate, residuals_candidate, _ = history.get_centered_entries(
        center_info=center_info,
        index=indices_not_min,
    )

    initial_residual_model = {"intercepts": history.get_best_residuals()}
    residual_model = update_initial_residual_model(
        initial_residual_model, x_candidate, residuals_candidate
    )
    main_model = update_main_from_residual_model(
        residual_model, multiply_square_terms_with_residuals=False
    )

    x_accepted = history.get_best_x()
    gradient_norm = np.linalg.norm(main_model["linear_terms"])
    gradient_norm *= delta

    valid = True
    reason = True
    n_modelpoints = n + 1

    last_model_indices = np.zeros(n_maxinterp, dtype=int)

    while reason is True:
        niter += 1

        # Solve the subproblem min{Q(s): ||s|| <= 1.0}
        result_sub = solve_subproblem(
            solution=x_accepted,
            delta=delta,
            main_model=main_model,
            ftol=ftol_sub,
            xtol=xtol_sub,
            gtol=gtol_sub,
            solver=solver_sub,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        qmin = -result_sub.fun
        x_candidate = x_accepted + result_sub.x * delta
        residuals_candidate = criterion(x_candidate)
        history.add_entries(x_candidate, residuals_candidate)

        rho = (history.get_critvals(accepted_index) - history.get_critvals(-1)) / qmin

        if (rho >= eta1) or (rho > eta0 and valid is True):
            residual_model["intercepts"] = history.get_residuals(index=accepted_index)
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

        # The model is deemend "not valid" if it has less than n model points.
        # Otherwise, if the model has n points it is considered "valid" or
        # "fully linear".
        # Note: valid is True in first iteration
        if valid is False:
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
                history, model_indices = add_points_to_make_main_model_fully_linear(
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

        # Update the trust region radius
        delta_old = delta
        norm_x_sub = np.sqrt(np.sum(result_sub.x ** 2))

        if rho >= eta1 and norm_x_sub > 0.5 * delta:
            delta = min(delta * gamma1, delta_max)
        elif valid is True:
            delta = max(delta * gamma0, delta_min)

        # Compute the next interpolation set
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
                # Model not valid. Add geometry points
                (
                    history,
                    model_indices,
                    n_modelpoints,
                ) = add_points_to_make_main_model_fully_linear(
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

        model_indices[1 : n_modelpoints + 1] = model_indices[:n_modelpoints]
        n_modelpoints += 1
        model_indices[0] = accepted_index

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
            n_maxinterp=n_maxinterp,
            n_modelpoints=n_modelpoints,
        )

        center_info = {"x": x_accepted, "radius": delta_old}
        interpolation_set = history.get_centered_xs(
            center_info, index=model_indices[:n_modelpoints]
        )

        f_interpolated = interpolate_f(
            history=history,
            interpolation_set=interpolation_set,
            residual_model=residual_model,
            model_indices=model_indices,
            n_modelpoints=n_modelpoints,
            n_maxinterp=n_maxinterp,
        )

        coefficients_residual_model = get_coefficients_residual_model(
            x_sample_monomial_basis=x_sample_monomial_basis,
            monomial_basis=monomial_basis,
            basis_null_space=basis_null_space,
            lower_triangular=lower_triangular,
            f_interpolated=f_interpolated,
            n_modelpoints=n_modelpoints,
        )

        residual_model["intercepts"] = history.get_residuals(index=accepted_index)
        residual_model = update_residual_model(
            residual_model=residual_model,
            coefficients_to_add=coefficients_residual_model,
            delta=delta,
            delta_old=delta_old,
        )

        main_model = update_main_from_residual_model(residual_model)

        gradient_norm = np.linalg.norm(main_model["linear_terms"])
        gradient_norm *= delta

        if gradient_norm < gtol:
            reason = False

        if niter > maxiter:
            reason = False

        # Test for repeated model
        if n_modelpoints == last_n_modelpoints:
            same = True
        else:
            same = False

        for i in range(n_modelpoints):
            if same:
                if model_indices[i] == last_model_indices[i]:
                    same = True
                else:
                    same = False
            last_model_indices[i] = model_indices[i]

        last_n_modelpoints = n_modelpoints
        if (same is True) and (delta == delta_old):
            # Identical model used in successive iterations
            reason = False

    result_dict = {
        "solution_x": history.get_best_x(),
        "solution_criterion": history.get_best_residuals(),
        "history_x": history.get_xs(),
        "history_criterion": history.get_residuals(),
        "n_iterations": niter,
        "message": "Under development.",
    }

    return result_dict

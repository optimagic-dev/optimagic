from functools import partial

import numpy as np
from estimagic.optimization.history import LeastSquaresHistory
from estimagic.optimization.pounders_auxiliary import add_more_points
from estimagic.optimization.pounders_auxiliary import calc_first_and_second_derivative
from estimagic.optimization.pounders_auxiliary import compute_criterion_norm
from estimagic.optimization.pounders_auxiliary import find_affine_points
from estimagic.optimization.pounders_auxiliary import get_approximation_error
from estimagic.optimization.pounders_auxiliary import get_params_quadratic_model
from estimagic.optimization.pounders_auxiliary import improve_model
from estimagic.optimization.pounders_auxiliary import solve_subproblem
from estimagic.optimization.pounders_auxiliary import update_center
from estimagic.optimization.pounders_auxiliary import update_gradient_and_hessian


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
):
    algorithm_info = {
        "primary_criterion_entry": "root_contributions",
        "parallelizes": False,
        "needs_scaling": True,
        "name": "pounders",
    }
    criterion = partial(
        criterion_and_derivative, algorithm_info=algorithm_info, task="criterion"
    )
    n_errors = len(
        criterion_and_derivative.keywords["first_criterion_evaluation"]["output"][
            "root_contributions"
        ]
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
        n_obs=n_errors,
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
    )

    return result_sub


def internal_solve_pounders(
    x0,
    n_obs,
    criterion,
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
    maxiter,
    gtol,
    ftol_sub,
    xtol_sub,
    gtol_sub,
    solver_sub,
    lower_bounds,
    upper_bounds,
):
    """Minimize criterion function using POUNDERS.

    Args:
        x0 (np.ndarray): Initial guess of the parameter vector. Starting points.
        n_obs (int): Number of observations/evaluation points.
        criterion (callable): Criterion function to be minimized.
        delta (float): Delta, initial trust-region radius.
        delta_min (float): Minimum value for delta.
        delta_max (float): Maximum value for delta.
        gamma0 (float): Gamma_0.
        gamma1 (float): Gamma_1.
        theta1 (float): Theta_1.
        theta2 (float): Theta_2.
        eta0 (float): Eta_0.
        eta1 (float): Eta_1.
        c1 (float): C_1. Equal to sqrt(*nparams*) by default.
        c2 (int)): C_2. Equal to 10 by default.
        maxiter (int): Maximum number of iterations. If reached, terminate.
        gtol_sub (float): Gradient norm used in the subproblem.
        solver_sub (str): Bound-constraint minimizer for the subproblem.
            Currently, three solvers from the scipy library are supported.
            - "trust-constr"
            - "L-BFGS-B"
            - "SLSQP"
        lower_bounds (np.ndarray): Lower bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.

    Returns:
        Tuple:
        - solution (np.ndarray): Solution vector.
        - gradient (np.ndarray): Gradient associated with the solution vector.
    """
    centered_criterion = partial(centered_criterion_template, criterion=criterion)

    history = LeastSquaresHistory()

    n = x0.shape[0]
    n_maxinterp = 2 * n + 1
    model_indices = np.zeros(n_maxinterp, dtype=int)

    history_x = np.zeros((maxiter * 2, n))  ###
    history_criterion = np.zeros((maxiter * 2, n_obs))  ###
    history_criterion_norm = np.zeros(maxiter * 2)  ###

    hessian = np.zeros((n_obs, n, n))

    last_n_modelpoints = 0
    niter = 0

    if lower_bounds is not None and upper_bounds is not None:
        if np.max(x0 + delta - upper_bounds) > 1e-10:
            raise ValueError("Starting points + delta > upper bounds.")

    min_criterion = criterion(x0)
    min_criterion_norm = compute_criterion_norm(min_criterion)

    history_x[0] = x0  ###
    history_criterion[0, :] = min_criterion  ###
    history_criterion_norm[0] = min_criterion_norm  ###

    index_min_x = 0

    # Increment parameters separately by delta
    for i in range(n):
        x1 = np.copy(x0)
        x1[i] = x1[i] + delta

        history_x[i + 1, :] = x1
        history_criterion[i + 1, :] = criterion(x1)
        history_criterion_norm[i + 1] = compute_criterion_norm(
            criterion_value=history_criterion[i + 1, :]
        )

        if history_criterion_norm[i + 1] < min_criterion_norm:
            min_criterion_norm = history_criterion_norm[i + 1]
            index_min_x = i + 1

    min_x = history_x[index_min_x, :]
    min_criterion = history_criterion[index_min_x, :]

    # Center around new trust-region and normalize to [-1, 1]
    indices_not_min = [i for i in range(n + 1) if i != index_min_x]
    xk = (history_x[indices_not_min, :] - min_x) / delta
    finite_difference = history_criterion[indices_not_min, :] - min_criterion

    # Determine the initial quadratic model
    gradient = np.linalg.solve(xk, finite_difference)

    first_derivative = np.dot(gradient, min_criterion)
    second_derivative = np.dot(gradient, gradient.T)
    gradient_norm = np.linalg.norm(first_derivative)
    gradient_norm *= delta

    valid = True
    reason = True
    n_history = n + 1
    n_modelpoints = n + 1

    last_model_indices = np.zeros(n_maxinterp, dtype=int)

    while reason is True:
        niter += 1

        # Solve the subproblem min{Q(s): ||s|| <= 1.0}
        result_sub = solve_subproblem(
            solution=history_x[index_min_x, :],
            delta=delta,
            first_derivative=first_derivative,
            second_derivative=second_derivative,
            ftol=ftol_sub,
            xtol=xtol_sub,
            gtol=gtol_sub,
            solver=solver_sub,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        qmin = -result_sub.fun
        xplus = min_x + result_sub.x * delta

        history_x[n_history, :] = xplus
        history_criterion[n_history, :] = criterion(history_x[n_history, :])
        history_criterion_norm[n_history] = compute_criterion_norm(
            criterion_value=history_criterion[n_history, :]
        )
        rho = (
            history_criterion_norm[index_min_x] - history_criterion_norm[n_history]
        ) / qmin

        n_history += 1

        if (rho >= eta1) or (rho > eta0 and valid is True):
            (
                min_x,
                min_criterion,
                gradient,
                min_criterion_norm,
                first_derivative,
                index_min_x,
            ) = update_center(
                xplus=xplus,
                min_x=min_x,
                history_x=history_x,
                delta=delta,
                min_criterion=min_criterion,
                gradient=gradient,
                history_criterion_norm=history_criterion_norm,
                hessian=hessian,
                first_derivative=first_derivative,
                second_derivative=second_derivative,
                n_history=n_history,
            )

        # Evaluate at a model improving point if necessary
        # Note: valid is True in first iteration
        if valid is False:
            (
                model_improving_points,
                model_indices,
                n_modelpoints,
                project_x_onto_null,
            ) = find_affine_points(
                history_x=history_x,
                min_x=min_x,
                model_improving_points=np.zeros((n, n)),
                project_x_onto_null=False,
                delta=delta,
                theta1=theta1,
                c=c1,
                model_indices=model_indices,
                n=n,
                n_modelpoints=0,
                n_history=n_history,
            )

            if n_modelpoints < n:
                add_all_points = 1
                (
                    history_x,
                    history_criterion,
                    history_criterion_norm,
                    model_indices,
                    n_modelpoints,
                    n_history,
                ) = improve_model(
                    history_x=history_x,
                    history_criterion=history_criterion,
                    history_criterion_norm=history_criterion_norm,
                    first_derivative=first_derivative,
                    second_derivative=second_derivative,
                    model_improving_points=model_improving_points,
                    model_indices=model_indices,
                    index_min_x=index_min_x,
                    n_modelpoints=n_modelpoints,
                    add_all_points=add_all_points,
                    n=n,
                    n_history=n_history,
                    delta=delta,
                    criterion=criterion,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )

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
            history_x=history_x,
            min_x=min_x,
            model_improving_points=np.zeros((n, n)),
            project_x_onto_null=False,
            delta=delta,
            theta1=theta1,
            c=c1,
            model_indices=model_indices,
            n=n,
            n_modelpoints=0,
            n_history=n_history,
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
                history_x=history_x,
                min_x=min_x,
                model_improving_points=model_improving_points,
                project_x_onto_null=project_x_onto_null,
                delta=delta,
                theta1=theta1,
                c=c2,
                model_indices=model_indices,
                n=n,
                n_modelpoints=n_modelpoints,
                n_history=n_history,
            )

            if n > n_modelpoints:
                # Model not valid. Add geometry points
                add_all_points = n - n_modelpoints
                (
                    history_x,
                    history_criterion,
                    history_criterion_norm,
                    model_indices,
                    n_modelpoints,
                    n_history,
                ) = improve_model(
                    history_x=history_x,
                    history_criterion=history_criterion,
                    history_criterion_norm=history_criterion_norm,
                    first_derivative=first_derivative,
                    second_derivative=second_derivative,
                    model_improving_points=model_improving_points,
                    model_indices=model_indices,
                    index_min_x=index_min_x,
                    n_modelpoints=n_modelpoints,
                    add_all_points=add_all_points,
                    n=n,
                    n_history=n_history,
                    delta=delta,
                    criterion=criterion,
                    lower_bounds=lower_bounds,
                    upper_bounds=upper_bounds,
                )

        model_indices[1 : n_modelpoints + 1] = model_indices[:n_modelpoints]
        n_modelpoints += 1
        model_indices[0] = index_min_x

        (
            lower_triangular,
            basis_null_space,
            monomial_basis,
            interpolation_set,
            n_modelpoints,
        ) = add_more_points(
            history_x=history_x,
            min_x=min_x,
            model_indices=model_indices,
            index_min_x=index_min_x,
            delta=delta,
            c2=c2,
            theta2=theta2,
            n=n,
            n_maxinterp=n_maxinterp,
            n_modelpoints=n_modelpoints,
            n_history=n_history,
        )

        xk = (history_x[model_indices[:n_modelpoints]] - min_x) / delta_old

        approximation_error = get_approximation_error(
            xk=xk,
            hessian=hessian,
            history_criterion=history_criterion,
            min_criterion=min_criterion,
            gradient=gradient,
            model_indices=model_indices,
            n_modelpoints=n_modelpoints,
            n_obs=n_obs,
            n_maxinterp=n_maxinterp,
        )

        params_gradient, params_hessian = get_params_quadratic_model(
            lower_triangular=lower_triangular,
            basis_null_space=basis_null_space,
            monomial_basis=monomial_basis,
            interpolation_set=interpolation_set,
            approximation_error=approximation_error,
            n_modelpoints=n_modelpoints,
            n=n,
            n_obs=n_obs,
        )

        gradient, hessian = update_gradient_and_hessian(
            gradient=gradient,
            hessian=hessian,
            params_gradient=params_gradient,
            params_hessian=params_hessian,
            delta=delta,
            delta_old=delta_old,
        )

        min_criterion = history_criterion[index_min_x]
        min_criterion_norm = history_criterion_norm[index_min_x]
        first_derivative, second_derivative = calc_first_and_second_derivative(
            gradient=gradient, min_criterion=min_criterion, hessian=hessian
        )

        gradient_norm = np.linalg.norm(first_derivative)
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

    result_sub_dict = {
        "solution_x": history_x[index_min_x, :],
        "solution_criterion": history_criterion[index_min_x, :],
        "history_x": history_x[:n_history, :],
        "history_criterion": history_criterion[:n_history, :],
        "n_iterations": niter,
        "message": "Under development.",
    }

    return result_sub_dict


def centered_criterion_template(centered_x, center_info, criterion):
    x = centered_x * center_info["radius"] + center_info["x"]
    out = criterion(x) + center_info["residuals"]
    return out

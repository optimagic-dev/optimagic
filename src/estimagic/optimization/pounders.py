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
    history = LeastSquaresHistory()

    n = x0.shape[0]
    n_maxinterp = 2 * n + 1
    model_indices = np.zeros(n_maxinterp, dtype=int)

    residual_hessians = np.zeros((n_obs, n, n))

    last_n_modelpoints = 0
    niter = 0

    if lower_bounds is not None and upper_bounds is not None:
        if np.max(x0 + delta - upper_bounds) > 1e-10:
            raise ValueError("Starting points + delta > upper bounds.")

    residuals_accepted = criterion(x0)
    history.add_entries(x0, residuals_accepted)

    accepted_index = 0

    xs = []
    for i in range(n):
        x1 = np.copy(x0)
        x1[i] += delta
        xs.append(x1)

    residuals = [criterion(x) for x in xs]

    history.add_entries(xs, residuals)
    accepted_index = history.get_min_index()

    x_accepted, residuals_accepted, _ = history.get_entries(index=accepted_index)

    # Center around new trust-region and normalize to [-1, 1]
    indices_not_min = [i for i in range(n + 1) if i != accepted_index]

    center_info = {"x": x_accepted, "radius": delta, "residuals": residuals_accepted}
    xk, fdiff, _ = history.get_centered_entries(center_info, index=indices_not_min)

    # Determine the initial quadratic model
    residual_gradients = np.linalg.solve(xk, fdiff)  # should this be called

    main_gradient = np.dot(residual_gradients, residuals_accepted)
    main_hessian = np.dot(residual_gradients, residual_gradients.T)
    gradient_norm = np.linalg.norm(main_gradient)
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
            solution=x_accepted,
            delta=delta,
            first_derivative=main_gradient,
            second_derivative=main_hessian,
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

        # history.add_entries(x_candidate, residuals_candidate) # noqa: E800
        # ==================================================================================
        history_x = history.xs
        history_criterion = history.residuals
        history_criterion_norm = history.critvals
        # ==================================================================================

        history_x[n_history, :] = x_candidate
        history_criterion[
            n_history, :
        ] = residuals_candidate  # criterion(x_candidate)# history_x[n_history, :])
        history_criterion_norm[n_history] = compute_criterion_norm(residuals_candidate)

        rho = (
            history_criterion_norm[accepted_index] - history_criterion_norm[n_history]
        ) / qmin

        n_history += 1

        if (rho >= eta1) or (rho > eta0 and valid is True):
            (
                x_accepted,
                residuals_accepted,
                residual_gradients,
                _,  # remove this from the outputs of update_center
                main_gradient,
                accepted_index,
            ) = update_center(
                xplus=x_candidate,
                min_x=x_accepted,
                history_x=history_x,
                delta=delta,
                min_criterion=residuals_accepted,
                gradient=residual_gradients,
                history_criterion_norm=history_criterion_norm,
                hessian=residual_hessians,
                first_derivative=main_gradient,
                second_derivative=main_hessian,
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
                min_x=x_accepted,
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
                    first_derivative=main_gradient,
                    second_derivative=main_hessian,
                    model_improving_points=model_improving_points,
                    model_indices=model_indices,
                    index_min_x=accepted_index,
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
            min_x=x_accepted,
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
                min_x=x_accepted,
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
                    first_derivative=main_gradient,
                    second_derivative=main_hessian,
                    model_improving_points=model_improving_points,
                    model_indices=model_indices,
                    index_min_x=accepted_index,
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
        model_indices[0] = accepted_index

        (
            lower_triangular,
            basis_null_space,
            monomial_basis,
            interpolation_set,
            n_modelpoints,
        ) = add_more_points(
            history_x=history_x,
            min_x=x_accepted,
            model_indices=model_indices,
            index_min_x=accepted_index,
            delta=delta,
            c2=c2,
            theta2=theta2,
            n=n,
            n_maxinterp=n_maxinterp,
            n_modelpoints=n_modelpoints,
            n_history=n_history,
        )

        xk = (history_x[model_indices[:n_modelpoints]] - x_accepted) / delta_old

        approximation_error = get_approximation_error(
            xk=xk,
            hessian=residual_hessians,
            history_criterion=history_criterion,
            min_criterion=residuals_accepted,
            gradient=residual_gradients,
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

        residual_gradients, residual_hessians = update_gradient_and_hessian(
            gradient=residual_gradients,
            hessian=residual_hessians,
            params_gradient=params_gradient,
            params_hessian=params_hessian,
            delta=delta,
            delta_old=delta_old,
        )

        residuals_accepted = history_criterion[accepted_index]
        _ = history_criterion_norm[accepted_index]
        main_gradient, main_hessian = calc_first_and_second_derivative(
            gradient=residual_gradients,
            min_criterion=residuals_accepted,
            hessian=residual_hessians,
        )

        gradient_norm = np.linalg.norm(main_gradient)
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
        "solution_x": history_x[accepted_index, :],
        "solution_criterion": history_criterion[accepted_index, :],
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

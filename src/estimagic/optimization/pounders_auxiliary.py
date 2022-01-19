"""Auxiliary functions for the pounders algorithm."""
from functools import partial

import numpy as np
from scipy.linalg import qr_multiply
from scipy.optimize import Bounds
from scipy.optimize import minimize


def update_initial_residual_model(
    initial_residual_model, x_candidate, residuals_candidate
):
    """Update linear and square terms of the initial residual model.

    Args:
        initial_residual_model (dict): Dictionary containing the parameters
            of the initial residual model, including "intercepts",
            "linear_terms", and "square terms".
        x_candidate (np.ndarray): Vector of centered x candidates of shape (n,).
        residuals_candidate (np.ndarray): Array of the corresponding centered
            residuals of shape (n, nobs).

    Returns:
        residual_model_updated (dict): Dictionary containing the parameters of
            the residual model with update "linear_terms" and "square_terms".
    """
    residual_model_updated = initial_residual_model.copy()
    n, n_obs = x_candidate.shape[0], residuals_candidate.shape[1]

    residual_model_updated["linear_terms"] = np.linalg.solve(
        x_candidate, residuals_candidate
    )
    residual_model_updated["square_terms"] = np.zeros((n_obs, n, n))

    return residual_model_updated


def update_residual_model(residual_model, coefficients_to_add, delta, delta_old):
    """Update linear and square terms of the residual model.

    Args:
        residual_model (dict): Dictionary containing the parameters of the
            residual model, i.e. "intercepts", "linear_terms", and "square terms".
        coefficients_to_add (dict): Coefficients used for updating the
            parameters of the residual model.
        delta (float): Trust region radius of the current iteration.
        delta_old (float): Trust region radius of the previous iteration.

    Returns:
        residual_model_updated (dict): Dictionary containing the parameters of
            the residual model with update "linear_terms" and "square_terms".
    """
    residual_model_updated = residual_model.copy()

    residual_model_updated["linear_terms"] = (
        coefficients_to_add["linear_terms"]
        + (delta / delta_old) * residual_model["linear_terms"]
    )

    residual_model_updated["square_terms"] = (
        coefficients_to_add["square_terms"]
        + (delta / delta_old) ** 2 * residual_model["square_terms"]
    )

    return residual_model_updated


def update_main_from_residual_model(
    residual_model, multiply_square_terms_with_residuals=True
):
    """Update linear and square terms of the main model via the residual model.

    Args:
        residual_model (dict): Dictionary containing the parameters of the
            residual model, i.e. "intercepts", "linear_terms", and "square terms".
        multiply_square_terms_with_residuals (bool): Indicator whether we
            multiply the main model's "square terms" with residuals, i.e.
            the intercepts of the residual model.

    Returns:
        main_model (dict): Dictionary containing the updated parameters of the
        main model, i.e. "linear_terms" and "square terms".
    """
    intercepts_residual_model = residual_model["intercepts"]
    linear_terms_residual_model = residual_model["linear_terms"]

    linear_terms_main_model = np.dot(
        linear_terms_residual_model, intercepts_residual_model
    )
    square_terms_main_model = np.dot(
        linear_terms_residual_model, linear_terms_residual_model.T
    )

    if multiply_square_terms_with_residuals is True:
        # Multiply 3d array *square_terms_residual_model* with
        # 1d array *intercepts_residual_model* along axis 0 of the former.
        square_terms_residual_model = residual_model["square_terms"]

        dim_array = np.ones((1, square_terms_residual_model.ndim), int).ravel()
        dim_array[0] = -1

        intercepts_reshaped = intercepts_residual_model.reshape(dim_array)

        square_terms_main_model = square_terms_main_model + np.sum(
            intercepts_reshaped * square_terms_residual_model, axis=0
        )

    main_model = {
        "linear_terms": linear_terms_main_model,
        "square_terms": square_terms_main_model,
    }

    return main_model


def update_main_model_with_new_accepted_x(main_model, x_candidate):
    """Use accepted candidate to update the linear terms of the residual model.

    Args:
        main_model (dict): Dictionary containing the parameters of the
            main model, i.e. "linear_terms" and "square terms".
        x_candidate (np.ndarray): Vector of centered x candidates of shape (n,).

    Returns:
        main_model_updated (dict): Dictionary containing the parameters of the
            main model with updated "linear_terms".
    """
    main_model_updated = main_model.copy()

    main_model_updated["linear_terms"] = main_model["linear_terms"] + np.dot(
        main_model["square_terms"], x_candidate
    )

    return main_model_updated


def update_residual_model_with_new_accepted_x(residual_model, x_candidate):
    """Use accepted candidate to update residual model.

    Args:
        residual_model (dict): Dictionary containing the parameters of
            the residual model, i.e. <"intercepts", "linear_terms", and
            "square terms".
        x_candidate (np.ndarray): Vector of centered x candidates of shape (n,).

    Returns:
        residual_model (dict): Dictionary containing the parameters of the
            residual model with updated "intercepts" and "linear_terms".
    """
    residual_model_updated = residual_model.copy()

    residual_model_updated["intercepts"] = (
        residual_model["intercepts"]
        + np.dot(x_candidate, residual_model["linear_terms"])
        + 0.5 * np.dot(np.dot(x_candidate, residual_model["square_terms"]), x_candidate)
    )
    residual_model_updated["linear_terms"] = (
        residual_model["linear_terms"]
        + np.dot(residual_model["square_terms"], x_candidate).T
    )

    return residual_model_updated


def solve_subproblem(
    solution,
    delta,
    main_model,
    ftol,
    xtol,
    gtol,
    solver,
    lower_bounds,
    upper_bounds,
):
    """Solve the subproblem.

    Args:
        solution (np.ndarray): Current solution vector.
        delta (float): Current trust region radius.
        main_model (dict): Dictionary containing the parameters of the
            main model including "linear_terms" and "square terms".
        gtol (float): Gradient tolerance. Stopping criterion.
        solver (str): Minimizer used to solve the bound-constraint subproblem.
            Currently, three solvers from the scipy library are supported.
            - "trust-constr"
            - "lower_triangular-BFGS-B"
            - "Slower_triangularSQP"
        lower_bounds (np.ndarray): Lower bounds for the subproblem.
            Must have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds for the subproblem.
            Must have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.

    Returns:
        Dict[str, np.ndarray]: Result dictionary.
    """
    # Initial guess
    n = solution.shape[0]
    x0 = np.zeros(n)

    # Normalize bounds. If not specified, use unit cube [-1, 1]
    if lower_bounds is not None:
        lower_bounds = (lower_bounds - solution) / delta
        lower_bounds[lower_bounds < -1] = -1
    else:
        lower_bounds = -np.ones(n)

    if upper_bounds is not None:
        upper_bounds = (upper_bounds - solution) / delta
        upper_bounds[upper_bounds > 1] = 1
    else:
        upper_bounds = np.ones(n)

    # Check if bounds valid
    if np.max(lower_bounds - upper_bounds) > 1e-10:
        raise ValueError("Upper bounds < lower bounds in subproblem.")
    if np.max(lower_bounds - x0) > 1e-10:
        raise ValueError("Initial guess < lower bounds in subproblem.")
    if np.max(x0 - upper_bounds) > 1e-10:
        raise ValueError("Initial guess > upper bounds in subproblem.")

    bounds = Bounds(lower_bounds, upper_bounds)

    if solver == "trust-constr":
        solver_args = {"hess": "2-point"}
        options = {"xtol": xtol, "gtol": gtol}
    elif solver == "L-BFGS-B":
        solver_args = {}
        options = {"ftol": ftol, "gtol": gtol}
    elif solver == "SLSQP":
        solver_args = {}
        options = {"ftol": ftol}
    else:
        raise ValueError("Subproblem solver is not supported.")

    evaluate_main_model = partial(
        _evaluate_main_model,
        **main_model,
    )

    rslt = minimize(
        evaluate_main_model,
        x0,
        method=solver,
        jac=True,
        bounds=bounds,
        **solver_args,
        options=options,
    )

    # Test bounds post-solution
    if np.max(lower_bounds - rslt.x) > 1e-5:
        raise ValueError("Subproblem solution < lower bounds.")
    if np.max(rslt.x - upper_bounds) > 1e-5:
        raise ValueError("Subproblem solution > upper bounds.")

    return rslt


def find_affine_points(
    history,
    x_accepted,
    model_improving_points,
    project_x_onto_null,
    delta,
    theta1,
    c,
    model_indices,
    n_modelpoints,
):
    """Find affine points.

    Args:
        history (class): Class storing history of xs, residuals, and critvals.
        x_accepted (np.ndarray): Accepted solution vector of the subproblem.
            Shape (n,).
        model_improving_points (np.ndarray): Array of shape (n, n) including
            points to improve the main model, i.e. make the main model fully linear.
            If *project_x_onto_null* is False, it is an array filled with zeros.
        project_x_onto_null (int): Indicator whether to calculate the QR
            decomposition of *model_improving_points* and multiply it
             with vector *x_projected*.
        delta (float): Delta, current trust-region radius.
        theta1 (float): Threshold for adding the current x candidate to the model.
        c (float): Threshold for acceptance of the norm of our current x candidate.
        model_indices (np.ndarray): Indices related to the candidates of x
            that are currently in the main model. Shape (2 *n* + 1,).
        n_modelpoints (int): Current number of model points.

    Returns:
        Tuple:
        - model_improving_points (np.ndarray):  Array of shape (n, n) including
            points to improve the main model, i.e. make the main model fully linear.
        - model_indices (np.ndarray): Indices related to the candidates of x
            that are currently in the main model. Shape (2 *n* + 1,).
        - n_modelpoints (int): Current number of model points.
        - project_x_onto_null (int): Indicator whether to calculate the QR
            decomposition of *model_improving_points* and multiply it
            with vector *x_projected*.
            Relevant for next call of *find_affine_points()*.
    """
    n = x_accepted.shape[0]

    for i in range(history.get_n_fun() - 1, -1, -1):
        center_info = {"x": x_accepted, "radius": delta}
        x_candidate = history.get_centered_xs(center_info, index=i)
        candidate_norm = np.linalg.norm(x_candidate)

        x_projected = x_candidate

        if candidate_norm <= c:
            if project_x_onto_null is True:
                x_projected, _ = qr_multiply(model_improving_points, x_projected)

            proj = np.linalg.norm(x_projected[n_modelpoints:])

            # Add this index to the model
            if proj >= theta1:
                model_indices[n_modelpoints] = i
                model_improving_points[:, n_modelpoints] = x_candidate
                project_x_onto_null = True
                n_modelpoints += 1

            if n_modelpoints == n:
                break

    return model_improving_points, model_indices, n_modelpoints, project_x_onto_null


def add_points_to_make_main_model_fully_linear(
    history,
    main_model,
    model_improving_points,
    model_indices,
    x_accepted,
    n_modelpoints,
    delta,
    criterion,
    lower_bounds,
    upper_bounds,
    batch_evaluator,
    n_cores,
):
    """Add points until main model is fully linear.

    Args:
        history (class): Class storing history of xs, residuals, and critvals.
        main_model (dict): Dictionary containing the parameters of the main model,
            i.e. "linear_terms" and "square terms".
        model_improving_points (np.ndarray): Array of shape (n, n) including
            points to improve the main model.
        model_indices (np.ndarray): Indices of the candidates of x that are
            currently in the main model. Shape (2 * n + 1,).
        x_accepted (np.ndarray): Accepted solution vector of the subproblem.
            Shape (n,).
        n_modelpoints (int): Current number of model points.
        delta (float): Delta, current trust-region radius.
        criterion (callable): Criterion function.
        lower_bounds (np.ndarray): Lower bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds.
            Must have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.
        batch_evaluator (str or callable): Name of a pre-implemented batch evaluator
            (currently 'joblib' and 'pathos_mp') or Callable with the same interface
            as the estimagic batch_evaluators.
        n_cores (int): Number of processes used to parallelize the function
            evaluations.

    Returns:
        Tuple:
        - history (class): Class storing history of xs, residuals, and critvals.
        - model_indices (np.ndarray): Indices of the candidates of x that are
            currently in the main model. Shape (2 * n + 1,).
    """
    n = x_accepted.shape[0]

    current_history = history.get_n_fun()
    linear_terms = main_model["linear_terms"]

    x_candidate = np.zeros(n)
    x_candidates_list = []
    criterion_candidates_list = []

    model_improving_points, _ = qr_multiply(model_improving_points, np.eye(3))

    for i in range(n_modelpoints, n):
        change_direction = np.dot(model_improving_points[:, i], linear_terms)

        if change_direction > 0:
            model_improving_points[:, i] *= -1

        x_candidate = delta * model_improving_points[:, i] + x_accepted

        # Project into feasible region
        if lower_bounds is not None and upper_bounds is not None:
            x_candidate = np.median(
                np.stack([lower_bounds, x_candidate, upper_bounds]), axis=0
            )
        x_candidates_list.append(x_candidate)
        model_indices[i] = current_history + i - n_modelpoints

    criterion_candidates_list = batch_evaluator(
        criterion, arguments=x_candidates_list, n_cores=n_cores
    )

    history.add_entries(x_candidates_list, criterion_candidates_list)

    return history, model_indices


def get_interpolation_matrices_residual_model(
    history,
    x_accepted,
    model_indices,
    delta,
    c2,
    theta2,
    n_maxinterp,
    n_modelpoints,
):
    """Obtain matrices that will be used for interpolating the residual model.

    Args:
        history (class): Class storing history of xs, residuals, and critvals.
        x_accepted (np.ndarray): Accepted solution vector of the subproblem.
            Shape (n,).
        model_indices (np.ndarray): Indices of the candidates of x that are
            currently in the model. Shape (2 * n + 1,).
        delta (float): Delta, current trust-region radius.
        c2 (int): Threshold for acceptance of the norm of our current x candidate.
            Equal to 10 by default.
        theta2 (float): Threshold for adding the current x candidate to the model.
        n_maxinterp (int): Maximum number of interpolation points.
        n_modelpoints (int): Current number of model points.

    Returns:
        Tuple:
        - x_sample_monomial_basis (np.ndarray): Sample of xs used for
            building the monomial basis. When taken together, they
            form a basis for the linear space of quadratics in n
            variables.
            Shape(n_maxinterp, n * (n + 1) / 2).
        - monomial_basis (np.ndarray): Monomial basis for quadratic functions of x.
            Shape(n_maxinterp, n * (n + 1) / 2).
        - basis_null_space (np.ndarray): Basis for the null space of xs that
            form the monomial basis. Shape(n_maxinterp, len(n + 1 : n_modelpoints)).
        - lower_triangular (np.ndarray): Lower triangular matrix of xs that
            form the monomial basis. Shape(n_maxinterp, n * (n + 1) / 2).
        - n_modelpoints (int): Current number of model points.
    """
    n = x_accepted.shape[0]

    x_sample_monomial_basis = np.zeros((n_maxinterp, n + 1))
    x_sample_monomial_basis[:, 0] = 1
    monomial_basis = np.zeros((n_maxinterp, int(n * (n + 1) / 2)))

    center_info = {"x": x_accepted, "radius": delta}
    for i in range(n + 1):
        x_sample_monomial_basis[i, 1:] = history.get_centered_xs(
            center_info, index=model_indices[i]
        )
        monomial_basis[i, :] = _get_monomial_basis(x_sample_monomial_basis[i, 1:])

    # Now we add points until we have n_maxinterp starting with the most recent ones
    point = history.get_n_fun() - 1
    n_modelpoints = n + 1

    while (n_modelpoints < n_maxinterp) and (point >= 0):
        reject = False

        # Reject any points already in the model
        for i in range(n + 1):
            if point == model_indices[i]:
                reject = True
                break

        if reject is False:
            candidate_x = history.get_centered_xs(center_info, index=point)
            candidate_norm = np.linalg.norm(candidate_x)

            if candidate_norm > c2:
                reject = True

        if reject is True:
            point -= 1
            continue

        x_sample_monomial_basis[n_modelpoints, 1:] = history.get_centered_xs(
            center_info, index=point
        )
        monomial_basis[n_modelpoints, :] = _get_monomial_basis(
            x_sample_monomial_basis[n_modelpoints, 1:]
        )

        x_sample_full_with_zeros = np.zeros((n_maxinterp, n_maxinterp))
        x_sample_full_with_zeros[:n_maxinterp, : n + 1] = x_sample_monomial_basis

        lower_triangular_temporary, _ = qr_multiply(
            x_sample_full_with_zeros[: n_modelpoints + 1, :],
            monomial_basis.T[: int(n * (n + 1) / 2), : n_modelpoints + 1],
        )
        beta = np.linalg.svd(lower_triangular_temporary.T[n + 1 :], compute_uv=False)

        if beta[min(n_modelpoints - n, int(n * (n + 1) / 2)) - 1] > theta2:
            # Accept point
            model_indices[n_modelpoints] = point
            lower_triangular = lower_triangular_temporary

            n_modelpoints += 1

        point -= 1

    # Orthogonal basis for the null space of M, where M is the
    # sample of xs forming the monomial basis
    basis_null_space, _ = qr_multiply(
        x_sample_full_with_zeros[:n_modelpoints, :],
        np.eye(n_maxinterp)[:, :n_modelpoints],
    )
    basis_null_space = basis_null_space[:, n + 1 : n_modelpoints]

    if n_modelpoints == (n + 1):
        lower_triangular = np.zeros((n_maxinterp, int(n * (n + 1) / 2)))
        lower_triangular[:n, :n] = np.eye(n)

    return (
        x_sample_monomial_basis,
        monomial_basis,
        basis_null_space,
        lower_triangular,
        n_modelpoints,
    )


def interpolate_f(
    history,
    interpolation_set,
    residual_model,
    model_indices,
    n_modelpoints,
    n_maxinterp,
):
    """Interpolate f via the quadratic residual model.

    If the point x_k belongs to the interpolation set, one can show that
    c = f (x_k).

    Args:
        history (class): Class storing history of xs, residuals, and critvals.
        x_sample (np.ndarray): Vector of centered x sample that makes up the
            interpolation set. Shape (maxinterp, n).
        residual_model (dict): Dictionary containing the parameters of the
            residual model, i.e. "intercepts", "linear_terms", and "square terms".
        model_indices (np.ndarray): Indices of the candidates of x that are
            currently in the model. Shape (2 *n* + 1,).
        n_modelpoints (int): Current number of model points.
        n_maxinterp (int): Maximum number of interpolation points.

    Returns:
        (np.ndarray): Interpolated function f. Array of shape (n_obs,).
    """
    n_obs = history.get_residuals(index=-1).shape[0]
    f_interpolated = np.zeros((n_maxinterp, n_obs), dtype=np.float64)

    for j in range(n_obs):
        x_square_terms = np.dot(
            interpolation_set, residual_model["square_terms"][j, :, :]
        )

        for i in range(n_modelpoints):
            center_info = {"residuals": residual_model["intercepts"]}
            residuals = history.get_centered_residuals(
                center_info, index=model_indices[i]
            )

            f_interpolated[i, j] = (
                residuals[j]
                - np.dot(residual_model["linear_terms"][:, j], interpolation_set[i, :])
                - 0.5 * np.dot(x_square_terms[i, :], interpolation_set[i, :])
            )

    return f_interpolated


def get_coefficients_residual_model(
    lower_triangular,
    basis_null_space,
    monomial_basis,
    x_sample_monomial_basis,
    f_interpolated,
    n_modelpoints,
):
    """Computes the coefficients of the quadratic residual model.

    The residual model:

        Q(x) = c + g'x + 0.5 x G x'

    satisfies the interpolation conditions Q(X[:,j]) = f(j)
    for j= 1,..., m and with a Hessian matrix of least Frobenius norm.

    Args:
        x_sample_monomial_basis (np.ndarray): Sample of xs used for
            building the monomial basis. When taken together, they
            form a basis for the linear space of quadratics in n
            variables.
            Shape(n_maxinterp, n * (n + 1) / 2).
        monomial_basis (np.ndarray): Monomial basis for quadratic functions of x.
            Shape(n_maxinterp, n * (n + 1) / 2).
        basis_null_space (np.ndarray): Basis for the null space of xs that
            form the monomial basis. Shape(n_maxinterp, len(n + 1 : n_modelpoints)).
        lower_triangular (np.ndarray): Lower triangular matrix of xs that
            form the monomial basis. Shape(n_maxinterp, n * (n + 1) / 2).
        f_interpolated (np.ndarray): Interpolated criterion function f.
            Shape (n_maxinterp, n_obs).
        n_modelpoints (int): Current number of model points.

    Returns:
        (dict): Coefficients for updating the "linear_terms" and "square_terms"
            of the residual model.
    """
    n = int((f_interpolated.shape[0] - 1) / 2)
    n_obs = f_interpolated.shape[1]

    params_gradient = np.zeros((n_obs, n))
    params_hessian = np.zeros((n_obs, n, n))
    lower_triangular = lower_triangular[:, n + 1 : n_modelpoints]

    if n_modelpoints == (n + 1):
        omega = np.zeros(n)
        beta = np.zeros(int(n * (n + 1) / 2))
    else:
        lower_triangular_square = np.dot(lower_triangular.T, lower_triangular)

    for k in range(n_obs):
        if n_modelpoints != (n + 1):
            lower_triangular_omega = np.dot(
                basis_null_space[:n_modelpoints, :].T,
                f_interpolated[:n_modelpoints, k],
            )
            omega = np.linalg.solve(
                np.atleast_2d(lower_triangular_square),
                np.atleast_1d(lower_triangular_omega),
            )

            beta = np.dot(np.atleast_2d(lower_triangular), omega)

        rhs = f_interpolated[:n_modelpoints, k] - np.dot(
            monomial_basis[:n_modelpoints, :], beta
        )

        alpha = np.linalg.solve(x_sample_monomial_basis[: n + 1, : n + 1], rhs[: n + 1])
        params_gradient[k, :] = alpha[1 : (n + 1)]

        num = 0
        for i in range(n):
            params_hessian[k, i, i] = beta[num]
            num += 1
            for j in range(i + 1, n):
                params_hessian[k, j, i] = beta[num] / np.sqrt(2)
                params_hessian[k, i, j] = beta[num] / np.sqrt(2)
                num += 1

    coefficients_to_add = {
        "linear_terms": params_gradient.T,
        "square_terms": params_hessian,
    }

    return coefficients_to_add


def _evaluate_main_model(
    x,
    linear_terms,
    square_terms,
):
    """Evaluate the criterion and derivative of the main model.

    Args:
        x (np.ndarray): Parameter vector of zeros.
        linear_terms (np.ndarray): Linear terms of the main model of shape (n,).
        square_terms (np.ndarray): Square terms of the main model of shape (n, n).

    Returns:
        Tuple:
        - criterion (float): Criterion value of the main model.
        - derivative (np.ndarray): Derivative of the main model of shape (n,).
    """
    criterion = np.dot(linear_terms, x) + 0.5 * np.dot(np.dot(x, square_terms), x)
    derivative = linear_terms + np.dot(square_terms, x)

    return criterion, derivative


def _get_monomial_basis(x):
    """Get the monomial basis (basis for quadratic functions) of x.

    Monomial basis = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n) ...
        ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n)^2]

    Args:
        x (np.ndarray): Parameter vector of shape (n,).

    Returns:
        (np.ndarray): Monomial basis of x wof shape (n * (n + 1) / 2,).
    """
    n = x.shape[0]
    monomial_basis = np.zeros(int(n * (n + 1) / 2))

    j = 0
    for i in range(n):
        monomial_basis[j] = 0.5 * x[i] ** 2
        j += 1

        for k in range(i + 1, n):
            monomial_basis[j] = x[i] * x[k] / np.sqrt(2)
            j += 1

    return monomial_basis

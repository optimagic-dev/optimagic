from functools import partial

import numpy as np
from scipy.linalg import qr_multiply
from scipy.optimize import Bounds
from scipy.optimize import minimize


def compute_criterion_norm(criterion_value):
    """Returns norm of the criterion function value.
    Args:
        criterion_value (np.ndarray): Value of the criterion function.
    Returns:
        (float): Norm of the criterion function.
    """
    return np.dot(criterion_value, criterion_value)


def update_center(
    x1,
    residuals_accepted,
    residual_gradients,
    residual_hessians,
    main_gradient,
    main_hessian,
):
    """Update center."""
    qk = (
        residuals_accepted
        + np.dot(x1, residual_gradients)
        + 0.5 * np.dot(np.dot(x1, residual_hessians), x1)
    )
    residual_gradients = residual_gradients + np.dot(residual_hessians, x1).T
    main_gradient = main_gradient + np.dot(main_hessian, x1)

    return (
        qk,
        residual_gradients,
        main_gradient,
    )


def calc_first_and_second_derivative(gradient, min_criterion, hessian):
    """Calculate first and second derivative of the criterion.

    Args:
        gradient (np.ndarray): Difference between the criterion function values
        and *min_criterion*. Shape (*n*, *n_obs*)
        min_criterion (np.ndarray): Values of criterion function associated with
            parameter vector x that yields the lowest criterion function norm.
        hessian (np.ndarray): Hessian matrix. Shape (*n_obs*, *n*, *n*).

    Returns:
        Tuple:
        - first_derivative (np.ndarray): Residuals of the Jacobian. Shape (*n*,).
        - second_derivative (np.ndarray): Residuals of the Hessian. Shape (*n*, *n*).
    """
    first_derivative = np.dot(gradient, min_criterion)

    second_derivative = np.dot(gradient, gradient.T)

    dim_array = np.ones((1, hessian.ndim), int).ravel()
    dim_array[0] = -1
    min_criterion_reshaped = min_criterion.reshape(dim_array)

    second_derivative = second_derivative + np.sum(
        min_criterion_reshaped * hessian, axis=0
    )

    return first_derivative, second_derivative


def solve_subproblem(
    solution,
    delta,
    first_derivative,
    second_derivative,
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
        first_derivative (np.ndarray): Residuals of the Jacobian. Shape (*n*,).
        second_derivative (np.ndarray): Residuals of the Hessian. Shape (*n*, *n*).
        gtol (float): Gradient tolerance. Stopping criterion.
        solver (str): candidate_xinimizer used to solve the bound-constraint subproblem.
            Currently, three solvers from the scipy library are supported.
            - "trust-constr"
            - "lower_triangular-BFGS-B"
            - "Slower_triangularSQP"
        lower_bounds (np.ndarray): lower_triangularower bounds for the subproblem.
            candidate_xust have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds for the subproblem.
            candidate_xust have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.

    Returns:
        Dict[str, np.ndarray]: Result dictionary.
    """
    # Initial guess
    n = first_derivative.shape[0]
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

    evaluate_subproblem = partial(
        _criterion_and_derivative_subproblem,
        first_derivative=first_derivative,
        second_derivative=second_derivative,
    )

    rslt = minimize(
        evaluate_subproblem,
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
    n,
    n_modelpoints,
):
    """Find affine points.

    Args:
        history_x (np.ndarray): Array storing all candidates of the parameter vector.
        x_accepted (np.ndarray): Values of parameter vector x that yield the lowest
            criterion function norm.
        model_improving_points (np.ndarray): Q matrix.
        project_x_onto_null (int): Indicator whether to calculate the QR
            decomposition of *model_improving_points* and multiply
            *model_improving_points* with vector *xk_plus*.
        delta (float): Delta, current trust-region radius.
        theta1 (float): Theta_1.
        c (float): C.
        model_indices (np.ndarray): Indices related to *history_x*, i.e. the
            candidates of x that are currently in the model. Shape (2 *n* + 1,).
        n (int): Number of parameters.
        n_modelpoints (int): Current number of model points.

    Returns:
        Tuple:
        - model_improving_points (np.ndarray): Q matrix.
        - model_indices (np.ndarray):
        - n_modelpoints (int): Current number of model points.
        - project_x_onto_null (int): Indicator whether to calculate the QR
            decomposition of *model_improving_points* and multiply
            *model_improving_points* with vector *xk_plus*.
            Relevant for next call of *find_nearby_points*.
    """
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


def improve_model(
    history,
    first_derivative,
    second_derivative,
    model_improving_points,
    model_indices,
    accepted_index,
    n_modelpoints,
    add_all_points,
    n,
    delta,
    criterion,
    lower_bounds,
    upper_bounds,
):
    """Improve the model.

    Args:
        history_x (np.ndarray): Array storing all candidates of the parameter
            vector. Shape (1000, *n*).
        history_criterion (np.ndarray): Array storing all evaluations of the criterion
            function. Shape(1000, *n_obs*).
        history_criterion_norm (np.ndarray): Array storing norm of the criterion
            function. Shape (1000,).
        first_derivative (np.ndarray): Residuals of the Jacobian. Shape (*n*,).
        second_derivative (np.ndarray): Residuals of the Hessian. Shape (*n*, *n*).
        model_improving_points (np.ndarray): Q matrix.
        model_indices (np.ndarray): Indices related to *history_x*, i.e. the
            candidates of x that are currently in the model. Shape (2 *n* + 1,).
        accepted_index (int): Index in *history_x* associated with the parameter vector
            that yields the lowest criterion function norm.
        n_modelpoints (int): Current number of model points.
        add_all_points (int): If equal to 0, add points. Else, don't.
        n (int): Number of parameters.
        delta (float): Delta, current trust-region radius.
        criterion (callable): Criterion function.
        lower_bounds (np.ndarray): lower_triangularower bounds.
            candidate_xust have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds.
            candidate_xust have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.

    Returns:
        Tuple:
        - history_x (np.ndarray): Array storing all candidates of the parameter
            vector. Shape (1000, *n*).
        - history_criterion (np.ndarray): Array storing all evaluations of the criterion
            function. Shape(1000, *n_obs*).
        - history_criterion_norm (np.ndarray): Array storing norm of the
            criterion function. Shape (1000,)
        - n_modelpoints (int): Current number of model points.
    """
    min_index_internal = 0
    minvalue = np.inf
    work = np.zeros(3)

    model_improving_points, _ = qr_multiply(model_improving_points, np.eye(3))

    for i in range(n_modelpoints, n):
        dp = np.dot(model_improving_points[:, i], first_derivative)

        # Go into other direction
        if dp > 0:
            model_improving_points[:, i] *= -1

        first_derivative_new = first_derivative + 0.5 * np.dot(
            second_derivative, model_improving_points[:, i]
        )
        work[i] = np.dot(model_improving_points[:, i], first_derivative_new)

        if (i == n_modelpoints) or (work[i] < minvalue):
            min_index_internal = i
            minvalue = work[i]

        if add_all_points != 0:
            (history, model_indices, n_modelpoints,) = _add_point(
                history=history,
                model_improving_points=model_improving_points,
                model_indices=model_indices,
                accepted_index=accepted_index,
                index=i,
                n_modelpoints=n_modelpoints,
                delta=delta,
                criterion=criterion,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )

    if add_all_points != 1:
        (history, model_indices, n_modelpoints,) = _add_point(
            history=history,
            model_improving_points=model_improving_points,
            model_indices=model_indices,
            accepted_index=accepted_index,
            index=min_index_internal,
            n_modelpoints=n_modelpoints,
            delta=delta,
            criterion=criterion,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    return (
        history,
        model_indices,
        n_modelpoints,
    )


def add_more_points(
    history,
    x_accepted,
    model_indices,
    delta,
    c2,
    theta2,
    n,
    n_maxinterp,
    n_modelpoints,
):
    """Add more points.
    Args:
        history_x (np.ndarray): Array storing all candidates of the parameter
            vector. Shape (1000, *n*).
        x_accepted (np.ndarray): Values of parameter vector x that yield the lowest
            criterion function norm.
        model_indices (np.ndarray): Indices related to *history_x*, i.e. the
            candidates of x that are currently in the model. Shape (2 *n* + 1,).
        accepted_index (int): Index in *history_x* associated with the parameter vector
            that yields the lowest criterion function norm.
        delta (float): Delta, current trust-region radius.
        c2 (int): C_2. Equal to 10 by default.
        theta2 (float): Theta_2.
        n (int): Number of parameters.
        n_maxinterp (int): candidate_xaximum number of interpolation points.
        n_modelpoints (int): Current number of model points.
    Returns:
        Tuple:
        - lower_triangular (np.ndarray): lower_triangular matrix.
            Shape(*n_maxinterp*, *n* (*n* + 1) / 2).
        - basis_null_space (np.ndarray): basis_null_space matrix.
            Shape(*n_maxinterp*, len(*n* + 1 : *n_modelpoints*)).
        - monomial_basis (np.ndarray): monomial_basis matrix.
            Shape(*n_maxinterp*, *n* (*n* + 1) / 2).
        - interpolation_set (np.ndarray): interpolation set.
            Shape(*n_maxinterp*, *n* + 1).
        - n_modelpoints (int): Current number of model points.
    """
    interpolation_set = np.zeros((n_maxinterp, n + 1))
    interpolation_set[:, 0] = 1
    monomial_basis = np.zeros((n_maxinterp, int(n * (n + 1) / 2)))

    for i in range(n + 1):
        center_info = {"x": x_accepted, "radius": delta}
        interpolation_set[i, 1:] = history.get_centered_xs(
            center_info, index=model_indices[i]
        )
        monomial_basis[i, :] = _get_basis_quadratic_function(x=interpolation_set[i, 1:])

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
            center_info = {"x": x_accepted, "radius": delta}
            candidate_x = history.get_centered_xs(center_info, index=point)
            candidate_norm = np.linalg.norm(candidate_x)

            if candidate_norm > c2:
                reject = True

        if reject is True:
            point -= 1
            continue

        center_info = {"x": x_accepted, "radius": delta}
        interpolation_set[n_modelpoints, 1:] = history.get_centered_xs(
            center_info, index=point
        )
        monomial_basis[n_modelpoints, :] = _get_basis_quadratic_function(
            x=interpolation_set[n_modelpoints, 1:]
        )

        interpolation_set_with_zeros = np.zeros((n_maxinterp, n_maxinterp))
        interpolation_set_with_zeros[:n_maxinterp, : n + 1] = interpolation_set

        lower_triangular_tmp, _ = qr_multiply(
            interpolation_set_with_zeros[: n_modelpoints + 1, :],
            monomial_basis.T[: int(n * (n + 1) / 2), : n_modelpoints + 1],
        )
        beta = np.linalg.svd(lower_triangular_tmp.T[n + 1 :], compute_uv=False)

        if beta[min(n_modelpoints - n, int(n * (n + 1) / 2)) - 1] > theta2:
            # Accept point
            model_indices[n_modelpoints] = point
            lower_triangular = lower_triangular_tmp

            n_modelpoints += 1

        point -= 1

    # Orthogonal basis for the null space of M, where M is the
    # interpolation set
    basis_null_space, _ = qr_multiply(
        interpolation_set_with_zeros[:n_modelpoints, :],
        np.eye(n_maxinterp)[:, :n_modelpoints],
    )
    basis_null_space = basis_null_space[:, n + 1 : n_modelpoints]

    if n_modelpoints == (n + 1):
        lower_triangular = np.zeros((n_maxinterp, int(n * (n + 1) / 2)))
        lower_triangular[:n, :n] = np.eye(n)

    return (
        lower_triangular,
        basis_null_space,
        monomial_basis,
        interpolation_set,
        n_modelpoints,
    )


def get_approximation_error(
    history,
    x_candidates,
    residual_model_accepted,
    hessian,
    gradient,
    model_indices,
    n_modelpoints,
    n_obs,
    n_maxinterp,
):
    """Calculate approximation error."""
    approximation_error = np.zeros((n_maxinterp, n_obs), dtype=np.float64)

    for j in range(n_obs):
        x_hessian = np.dot(x_candidates, hessian[j, :, :])

        for i in range(n_modelpoints):
            residuals = history.get_residuals(index=model_indices[i])
            approximation_error[i, j] = (
                residuals[j]
                - residual_model_accepted[j]
                - np.dot(gradient[:, j], x_candidates[i, :])
                - 0.5 * np.dot(x_hessian[i, :], x_candidates[i, :])
            )

    return approximation_error


def get_params_quadratic_model(
    lower_triangular,
    basis_null_space,
    monomial_basis,
    interpolation_set,
    approximation_error,
    n_modelpoints,
    n,
    n_obs,
):
    """Get parameters of quadratic model.

    Computes the parameters of the quadratic model Q(x) = c + g'x + 0.5 x G x'
    that satisfies the interpolation conditions Q(X[:,j]) = f(j)
    for j= 1,..., m and with a Hessian matrix of least Frobenius norm.
    Args:
        lower_triangular (np.ndarray): lower_triangular matrix.
            Shape(*n_maxinterp*, *n* (*n* + 1) / 2).
        basis_null_space (np.ndarray): basis_null_space matrix.
            Shape(:*n_modelpoints*, *n* + 1 : *n_modelpoints*).
        monomial_basis (np.ndarray): monomial_basis matrix.
            Shape(*n_maxinterp*, *n* + 1).
        interpolation_set (np.ndarray): Interpolation set.
            Shape(*n_maxinterp*, *n* (*n* + 1) / 2).
        approximation_error (np.ndarray): Approximation_error.
            Shape (*n_maxinterp*, *n_obs*).
        n_modelpoints (int): Current number of model points.
        n (int): Number of parameters.
        n_obs (int): Number of observations.

    Returns:
        Tuple:
        - params_gradient (np.ndarray): Jacobian of the quadratic model.
            Shape (*n_obs*, *n*).
        - params_hessian (np.ndarray): Hessian of the quadratic model.
            Shape (*n_obs*, *n*, *n*).
    """
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
            omega = np.dot(
                basis_null_space[:n_modelpoints, :].T,
                approximation_error[:n_modelpoints, k],
            )
            omega = np.linalg.solve(
                np.atleast_2d(lower_triangular_square), np.atleast_1d(omega)
            )

            beta = np.dot(np.atleast_2d(lower_triangular), omega)

        rhs = approximation_error[:n_modelpoints, k] - np.dot(
            monomial_basis[:n_modelpoints, :], beta
        )

        alpha = np.linalg.solve(interpolation_set[: n + 1, : n + 1], rhs[: n + 1])
        params_gradient[k, :] = alpha[1 : (n + 1)]

        num = 0
        for i in range(n):
            params_hessian[k, i, i] = beta[num]
            num += 1
            for j in range(i + 1, n):
                params_hessian[k, j, i] = beta[num] / np.sqrt(2)
                params_hessian[k, i, j] = beta[num] / np.sqrt(2)
                num += 1

    return params_gradient, params_hessian


def update_gradient_and_hessian(
    gradient, hessian, params_gradient, params_hessian, delta, delta_old
):
    """Update gradient and Hessian."""
    gradient_new = (delta / delta_old) * gradient + params_gradient.T
    hessian_new = (delta / delta_old) ** 2 * hessian + params_hessian

    return gradient_new, hessian_new


def _criterion_and_derivative_subproblem(
    x,
    first_derivative,
    second_derivative,
):
    """Returns the objective and gradient of the subproblem.

    Args:
        x (np.ndarray): Parameter vector.
        first_derivative (np.ndarray): Residuals of the Jacobian. Shape (*n*,).
        second_derivative (np.ndarray): Residuals of the Hessian. Shape (*n*, *n*).
    Returns:
        Tuple:
        - obj (float): Value of the objective function.
        - grad (np.ndarray): Gradient vector. Shape (*n*,).
    """
    criterion = np.dot(first_derivative, x) + 0.5 * np.dot(
        np.dot(x, second_derivative), x
    )
    derivative = first_derivative + np.dot(second_derivative, x)

    return criterion, derivative


def _get_basis_quadratic_function(x):
    """Evaluate phi.

    Phi = .5*[x(1)^2  sqrt(2)*x(1)*x(2) ... sqrt(2)*x(1)*x(n) ...
        ... x(2)^2 sqrt(2)*x(2)*x(3) .. x(n)^2]
    Args:
        x (np.ndarray): Parameter vector of shape (*n*,).
        n (int): Number of parameters.
    Returns:
        (np.ndarray): Monomial basis of shape (*n* (*n* + 1) / 2,)
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


def _add_point(
    history,
    model_improving_points,
    model_indices,
    accepted_index,
    index,
    n_modelpoints,
    delta,
    criterion,
    lower_bounds,
    upper_bounds,
):
    """Add point to the model

    Args:
        history_x (np.ndarray): Array storing all candidates of the parameter
            vector. Shape (1000, *n*).
        history_criterion (np.ndarray): Array storing all evaluations of the criterion
            function. Shape(1000, *n_obs*).
        history_criterion_norm (np.ndarray): Array storing norm of the criterion
            function. Shape (1000,).
        model_improving_points (np.ndarray): Q matrix containing the parameter
            vector to add to *history_x*. Shape (*n*, *n*).
        model_indices (np.ndarray): Indices related to *history_x*, i.e. the
            candidates of x that are currently in the model. Shape (2 *n* + 1,).
        accepted_index (int): Index in *history_x* associated with the parameter vector
            that yields the lowest criterion function norm.
        index (int): Index relating to the parameter vector in
            *model_improving_points* that is added to *history_x*.
        n_modelpoints (int): Current number of model points.
        delta (float): Delta, current trust-region radius.
        criterion (callable): Criterion function.
        lower_bounds (np.ndarray): lower_triangularower bounds.
            candidate_xust have same length as the initial guess of the
            parameter vector. Equal to -1 if not provided by the user.
        upper_bounds (np.ndarray): Upper bounds.
            candidate_xust have same length as the initial guess of the
            parameter vector. Equal to 1 if not provided by the user.

    Returns:
        Tuple:
        - history_x (np.ndarray): Array storing all candidates of the parameter
            vector. Shape (1000, *n*).
        - history_criterion (np.ndarray): Array storing all evaluations of the criterion
            function. Shape(1000, *n_obs*).
        - history_criterion_norm (np.ndarray): Array storing norm of the
            criterion function. Shape (1000,).
        - model_indices (np.ndarray): Indices related to *history_x*, i.e. the
            candidates of x that are currently in the model. Shape (2 *n* + 1,).
        - n_modelpoints (int): Current number of model points.
    """
    x_candidate = model_improving_points[:, index]
    x_candidate = delta * x_candidate + history.get_xs(index=accepted_index)

    # Project into feasible region
    if lower_bounds is not None and upper_bounds is not None:
        x_candidate = np.median(
            np.stack([lower_bounds, x_candidate, upper_bounds]), axis=0
        )

    history.add_entries(x_candidate, criterion(x_candidate))

    # Add new vector to the model
    model_indices[n_modelpoints] = history.get_n_fun()
    n_modelpoints += 1

    return (
        history,
        model_indices,
        n_modelpoints,
    )

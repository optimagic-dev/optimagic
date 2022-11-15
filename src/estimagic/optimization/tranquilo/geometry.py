from functools import partial

import numpy as np
from estimagic.optimization.subsolvers._trsbox_quadratic import minimize_trust_trsbox
from estimagic.optimization.tranquilo.models import ScalarModel
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import _get_effective_bounds
from estimagic.optimization.tranquilo.sample_points import (
    _map_from_feasible_trustregion,
)
from estimagic.optimization.tranquilo.sample_points import get_sampler


def get_geometry_checker_pair(
    checker, reference_sampler, n_params, n_simulations=200, bounds=None
):
    """Get a geometry checker.

    Args:
        checker (str or Dict[callable]): Name of a geometry checker method or a
            dictionary with entries 'quality_calculator' and 'cutoff_simulator'.
            - 'quality_calculator': A callable that takes as argument a sample and
            returns a measure on the quality of the geometry of the sample.
            - 'cutoff_simulator': A callable that takes as argument 'n_samples',
            'n_params', 'reference_sampler' and 'rng'.
        reference_sampler (str): Either "box" or "ball", corresponding to comparison
            samples drawn inside a box or a ball, respectively.
        n_params (int): Number of parameters.
        n_simulations (int): Number of simulations for the mean calculation.
        bounds (Bounds): The parameter bounds.

    Returns:
        callable: The sample quality calculator.
        callable: The quality cutoff simulator.

    """
    if reference_sampler not in {"box", "ball"}:
        raise ValueError("reference_sampler need to be either 'box' or 'ball'.")

    built_in_checker = {
        "d_optimality": {
            "quality_calculator": log_d_quality_calculator,
            "cutoff_simulator": log_d_cutoff_simulator,
        },
    }

    _checker = built_in_checker[checker]

    quality_calculator = partial(_checker["quality_calculator"], bounds=bounds)
    cutoff_simulator = partial(
        _checker["cutoff_simulator"],
        reference_sampler=reference_sampler,
        bounds=bounds,
        n_params=n_params,
        n_simulations=n_simulations,
    )
    return quality_calculator, cutoff_simulator


def log_d_cutoff_simulator(
    n_samples, rng, reference_sampler, bounds, n_params, n_simulations
):
    """Simulate the mean logarithm of the d-optimality criterion.

    Args:
        n_samples (int): Size of the sample.
        rng (np.random.Generator): The random number generator.
        reference_sampler (str): Either "box" or "ball", corresponding to comparison
            samples drawn inside a box or a ball, respectively.
        bounds (Bounds): The parameter bounds.
        n_params (int): Dimensionality of the sample.
        n_simulations (int): Number of simulations for the mean calculation.

    Returns:
        float: The simulated mean logarithm of the d-optimality criterion.

    """
    _sampler = get_sampler(reference_sampler, bounds)
    trustregion = TrustRegion(center=np.zeros(n_params), radius=1)
    sampler = partial(_sampler, trustregion=trustregion)
    raw = []
    for _ in range(n_simulations):
        x = sampler(n_points=n_samples, rng=rng)
        raw.append(log_d_quality_calculator(x, trustregion, bounds))
    out = np.nanmean(raw)
    return out


def log_d_quality_calculator(sample, trustregion, bounds):
    """Logarithm of the d-optimality criterion.

    For a data sample x the log_d_criterion is defined as log(det(x.T @ x)). If the
    determinant is zero the function returns -np.inf. Before computation the sample is
    mapped into unit space.

    Args:
        sample (np.ndarray): The data sample, shape = (n, p).
        trustregion (TrustRegion): NamedTuple with attributes center and radius.
        bounds (Bounds): The parameter bounds.

    Returns:
        np.ndarray: The criterion values, shape = (n, ).

    """
    effective_bounds = _get_effective_bounds(trustregion, bounds)
    points = _map_from_feasible_trustregion(sample, effective_bounds)
    n_samples, n_params = points.shape
    xtx = points.T @ points
    det = np.linalg.det(xtx / n_samples)
    out = n_params * np.log(n_samples) + np.log(det)
    return out


# =====================================================================================


def poisedness_constant(sample, lower_bounds, upper_bounds):
    """Calculate the poisedness constant of the points inside the trust region.

    Args:
        sample (np.ndarray): The data sample, shape = (n, p).
        bounds (Bounds): The parameter bounds.

    Returns:
        float: The maximum absolute criterion value.

    """
    n_samples, n_params = sample.shape
    trustregion = TrustRegion(center=np.zeros(n_params), radius=1)

    sample_with_center = np.row_stack((trustregion.center, sample))
    critval_max = -999

    for index in range(n_samples + 1):
        intercept, linear_terms, square_terms = get_lagrange_polynomial(
            sample_with_center,
            sample,
            index,
        )

        scalar_model = ScalarModel(
            intercept=intercept, linear_terms=linear_terms, square_terms=square_terms
        )
        xmax = maximize_absolute_value_trust_trsbox(
            scalar_model,
            trustregion,
            lower_bounds,
            upper_bounds,
        )

        critval_current = abs(_evaluate_scalar_model(xmax, scalar_model))

        if critval_current > critval_max:
            critval_max = critval_current

    return critval_max


def maximize_absolute_value_trust_trsbox(
    scalar_model,
    trustregion,
    lower_bounds,
    upper_bounds,
):
    """Maximize the absolute value of a Lagrange polynomial in a trust-region setting.

    Let a Lagrange polynomial of degree two be defined by:
        L(x) = c + g.T @ x + 0.5 x.T @ H @ x,

    where c, g, H denote the intercept, linear terms, and square terms of the
    scalar model, respectively.

    In order to maximize L(x), we maximize the absolute value of L(x) in a
    trust-region setting. I.e. we solve:

        max_x  abs(c + g.T @ x + 0.5 x.T @ H @ x)
            s.t. lower_bound <= x <= upper_bound
                 ||x|| <= trustregion_radius

    In order to find the solution x*, we both minimize and maximize
    the objective c + g.T @ x + 0.5 x.T @ H @ x.
    The resulting candidate vectors are then plugged into the objective function L(x)
    to check which one yields the largest absolute value of the Lagrange polynomial.

    Args:
        scalar_model (NamedTuple): Named tuple containing the parameters of the
            scalar surrogate model, i.e.:
            - ``intercept`` (float): Intercept of the scalar model.
            - ``linear_terms`` (np.ndarray): 1d array of shape (n,) with the linear
                terms of the mdoel.
            - ``square_terms`` (np.ndarray): 2d array of shape (n, n) containing
                the model's square terms
        trustregion (NamedTuple): Contains ``center`` (np.ndarray) and ``radius``
            (float). Used to center bounds.
        lower_bounds (np.ndarray): 1d array of shape (n,) with lower bounds
            for the parameter vector x.
        upper_bounds (np.ndarray): 1d array of shape (n,) with upper bounds
            for the parameter vector x.

    Returns:
        np.ndarray: Solution vector of shape (n,).

    """
    radius = trustregion.radius

    x_min = minimize_trust_trsbox(
        scalar_model.linear_terms,
        scalar_model.square_terms,
        radius,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )
    x_max = minimize_trust_trsbox(
        -scalar_model.linear_terms,
        -scalar_model.square_terms,
        radius,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    criterion_min = _evaluate_scalar_model(x_min, scalar_model)
    criterion_max = _evaluate_scalar_model(x_max, scalar_model)

    if abs(criterion_min) >= abs(criterion_max):
        x_out = x_min
    else:
        x_out = x_max

    return x_out


def get_lagrange_polynomial(
    sample,
    sample_without_xopt,
    index,
):
    n_samples, n_params = sample.shape

    center_index = 0
    center = sample[center_index]
    row_index = index - 1

    if n_samples == n_params + 1:
        if row_index >= 0:
            rhs = np.zeros(n_params)
            rhs[row_index] = 1.0
        else:
            rhs = -np.ones(n_params)
    elif n_samples == (n_params + 1) * (n_params + 2) // 2:
        if row_index >= 0:
            rhs = np.zeros(n_samples - 1)
            rhs[row_index] = 1.0
        else:
            rhs = -np.ones(n_samples - 1)
    else:
        rhs = np.zeros(n_samples + n_params - 1)
        if row_index >= 0:
            rhs[row_index] = 1.0
        else:
            rhs[: n_samples - 1] = -1.0

    interpolation_mat = build_interpolation_matrix(sample_without_xopt.T)
    coef, *_ = np.linalg.lstsq(interpolation_mat, rhs, rcond=None)

    intercept = 1.0 if index == center_index else 0.0

    if n_samples == n_params + 1:
        grad = coef
        hess = np.zeros((n_params, n_params))
    elif n_samples == (n_params + 1) * (n_params + 2) // 2:
        grad = coef[:n_params]
        hess = _build_symmetric_matrix_from_vector(coef[n_params:], n_params)
    else:
        grad = coef[n_samples - 1 :]
        fval_row_idx = np.arange(1, n_samples)

        hess = np.zeros((n_params, n_params))
        for i in range(n_samples - 1):
            dx = sample[fval_row_idx[i]] - center
            hess += coef[i] * np.outer(dx, dx)

    return intercept, grad, hess


def build_interpolation_matrix(sample):
    n_params, n_samples = sample.shape
    assert (
        n_params + 1 <= n_samples + 1 <= (n_params + 1) * (n_params + 2) // 2
    ), "npt must be in range [n+1, (n+1)(n+2)/2]"

    if n_samples == n_params:
        mat = sample.T
    elif n_samples + 1 == (n_params + 1) * (n_params + 2) // 2:
        mat = np.empty((n_samples, n_samples))
        mat[:, :n_params] = sample.T
        for i in range(n_samples):
            mat[i, n_params:] = _to_upper_triangular_vector(
                np.outer(sample[:, i], sample[:, i])
                - 0.5 * np.diag(np.square(sample[:, i]))
            )
    else:
        mat = np.zeros((n_samples + n_params, n_samples + n_params))
        for i in range(n_samples):
            for j in range(n_samples):
                mat[i, j] = 0.5 * np.dot(sample[:, i], sample[:, j]) ** 2
        mat[:n_samples, n_samples:] = sample.T
        mat[n_samples:, :n_samples] = sample

    return mat


def _build_symmetric_matrix_from_vector(entries, n):
    mat = np.empty((n, n))
    idx = -1

    for j in range(n):
        for i in range(j + 1):
            idx += 1
            mat[i, j] = entries[idx]
            mat[j, i] = entries[idx]

    return mat


def _to_upper_triangular_vector(triu):
    n = triu.shape[0]
    vec = np.empty(n * (n + 1) // 2)
    idx = -1

    for j in range(n):
        for i in range(j + 1):
            idx += 1
            vec[idx] = triu[i, j]

    return vec


def _evaluate_scalar_model(x, scalar_model):
    return (
        scalar_model.intercept
        + scalar_model.linear_terms.T @ x
        + 0.5 * x.T @ scalar_model.square_terms @ x
    )

from functools import partial

import numpy as np
from estimagic.optimization.subsolvers._trsbox_quadratic import minimize_trust_trsbox
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
        x = sampler(target_size=n_samples, rng=rng)
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
        trustregion:
        bounds:

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


def _evaluate_scalar_model(x, scalar_model):
    return (
        scalar_model.intercept
        + scalar_model.linear_terms.T @ x
        + 0.5 * x.T @ scalar_model.square_terms @ x
    )

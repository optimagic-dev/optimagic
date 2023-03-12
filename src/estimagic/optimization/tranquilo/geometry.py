from functools import partial

import numpy as np

from estimagic.optimization.tranquilo.region import Region
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
        bounds (Bounds): The parameter bounds. See module bounds.py.

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

    quality_calculator = _checker["quality_calculator"]
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
        bounds (Bounds): The parameter bounds. See module bounds.py.
        n_params (int): Dimensionality of the sample.
        n_simulations (int): Number of simulations for the mean calculation.

    Returns:
        float: The simulated mean logarithm of the d-optimality criterion.

    """
    _sampler = get_sampler(reference_sampler)
    trustregion = Region(center=np.zeros(n_params), radius=1, bounds=bounds)
    sampler = partial(_sampler, trustregion=trustregion)
    raw = []
    for _ in range(n_simulations):
        x = sampler(n_points=n_samples, rng=rng)
        raw.append(log_d_quality_calculator(x, trustregion))
    out = np.nanmean(raw)
    return out


def log_d_quality_calculator(sample, trustregion):
    """Logarithm of the d-optimality criterion.

    For a data sample x the log_d_criterion is defined as log(det(x.T @ x)). If the
    determinant is zero the function returns -np.inf. Before computation the sample is
    mapped into unit space.

    Args:
        sample (np.ndarray): The data sample, shape = (n, p).
        trustregion (Region): Trustregion. See module region.py.

    Returns:
        np.ndarray: The criterion values, shape = (n, ).

    """
    points = trustregion.map_to_unit(sample)
    n_samples, n_params = points.shape
    xtx = points.T @ points
    det = np.linalg.det(xtx / n_samples)
    out = n_params * np.log(n_samples) + np.log(det)
    return out

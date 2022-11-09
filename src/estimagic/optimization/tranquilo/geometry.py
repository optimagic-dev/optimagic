from functools import partial
from typing import NamedTuple

import numpy as np
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import _get_effective_bounds
from estimagic.optimization.tranquilo.sample_points import (
    _map_from_feasible_trustregion,
)
from estimagic.optimization.tranquilo.sample_points import get_sampler


class GeometryChecker(NamedTuple):
    quality_calculator: callable
    cutoff_simulator: callable


def get_geometry_checker(checker, reference_sampler, bounds, n_params):
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

    Returns:
        GeometryChecker: The geometry checker.

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

    out = GeometryChecker(
        quality_calculator=partial(_checker["quality_calculator"], bounds=bounds),
        cutoff_simulator=partial(
            _checker["cutoff_simulator"],
            reference_sampler=reference_sampler,
            bounds=bounds,
            n_params=n_params,
        ),
    )
    return out


def log_d_cutoff_simulator(
    n_samples, rng, reference_sampler, bounds, n_params, n_simulations=100
):
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

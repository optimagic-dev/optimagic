from functools import partial
from typing import NamedTuple

import numpy as np
from estimagic.optimization.tranquilo.options import TrustRegion
from estimagic.optimization.tranquilo.sample_points import get_sampler


class GeometryChecker(NamedTuple):
    quality_calculator: callable
    cutoff_simulator: callable


def get_geometry_checker(checker, reference_sampler, n_params):
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

    _cutoff_sampler = _get_cutoff_simulator_sampler(reference_sampler, n_params)

    out = GeometryChecker(
        quality_calculator=_checker["quality_calculator"],
        cutoff_simulator=partial(_checker["cutoff_simulator"], sampler=_cutoff_sampler),
    )
    return out


def _get_cutoff_simulator_sampler(reference_sampler, n_params):
    trustregion = TrustRegion(center=np.zeros(n_params), radius=1)
    sampler = get_sampler(reference_sampler, bounds=None)
    out = partial(sampler, trustregion=trustregion)
    return out


def log_d_cutoff_simulator(n_samples, rng, sampler, n_simulations=100):
    raw = []
    for _ in range(n_simulations):
        x = sampler(target_size=n_samples, rng=rng)
        raw.append(log_d_quality_calculator(x))
    out = np.nanmean(raw)
    return out


def log_d_quality_calculator(sample):
    """Logarithm of the d-optimality criterion.

    For a data sample x the log_d_criterion is defined as log(det(x.T @ x)). If the
    determinant is zero the function returns np.nan.

    Args:
        sample (np.ndarray): The data sample, shape = (n, p).

    Returns:
        np.ndarray: The criterion values, shape = (n, ).

    """
    n_samples, n_params = sample.shape
    xtx = sample.T @ sample
    det = np.linalg.det(xtx / n_samples)
    if det <= 0:
        out = np.nan
    else:
        out = n_params * np.log(n_samples) + np.log(det)
    return out

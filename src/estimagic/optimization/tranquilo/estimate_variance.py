"""Estimate the variance or covariance matrix of the noise in the objective function."""


import numpy as np

from estimagic.optimization.tranquilo.get_component import get_component
from estimagic.optimization.tranquilo.history import History
from estimagic.optimization.tranquilo.region import Region
from estimagic.optimization.tranquilo.options import VarianceEstimatorOptions


def get_variance_estimator(fitter, user_options):
    func_dict = {
        "classic": _estimate_variance_classic,
    }

    out = get_component(
        name_or_func=fitter,
        func_dict=func_dict,
        component_name="variance estimator",
        user_options=user_options,
        default_options=VarianceEstimatorOptions(),
    )

    return out


def _estimate_variance_classic(
    trustregion: Region,
    history: History,
    model_type: str,
    max_distance_factor: float,
    min_n_evals: int,
):
    all_indices = history.get_x_indices_in_region(
        trustregion._replace(radius=trustregion.radius * max_distance_factor)
    )

    n_evals = {idx: len(history.get_fvals(idx)) for idx in all_indices}

    # make sure we keep at least one sample from which we can estimate a variance
    cutoff = min(max(n_evals.values()), min_n_evals)

    valid_indices = [idx for idx in all_indices if n_evals[idx] >= cutoff]
    weights = np.array([n_ for idx, n_ in n_evals.items() if idx in valid_indices])
    weights = weights / weights.sum()

    if model_type == "scalar":
        samples = list(history.get_fvals(valid_indices).values())
        out = 0.0
        for weight, sample in zip(weights, samples):
            out += weight * np.var(sample, ddof=1)
    else:
        samples = list(history.get_fvecs(valid_indices).values())

        dim = samples[0].shape[1]
        out = np.zeros((dim, dim))
        for weight, sample in zip(weights, samples):
            out += weight * np.cov(sample, rowvar=False, ddof=1)

    return out

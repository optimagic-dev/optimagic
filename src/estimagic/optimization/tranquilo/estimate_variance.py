"""Estimate the variance or covariance matrix of the noise in the objective function."""

import numpy as np

from estimagic.optimization.tranquilo.get_component import get_component


def get_variance_estimator(fitter, user_options):
    func_dict = {
        "unweighted": _estimate_variance_unweighted,
    }

    default_options = {
        "target_sample_size": 25,
        "max_n_states": 5,
    }

    out = get_component(
        name_or_func=fitter,
        func_dict=func_dict,
        component_name="variance estimator",
        user_options=user_options,
        default_options=default_options,
    )

    return out


def _estimate_variance_unweighted(
    history, states, model_type, target_sample_size, max_n_states
):
    if model_type == "scalar":
        out = _unweighted_scalar(history, states, target_sample_size, max_n_states)
    elif model_type == "vector":
        out = _unweighted_vector(history, states, target_sample_size, max_n_states)
    else:
        raise ValueError("model_type must be scalar or vector.")

    return out


def _unweighted_scalar(history, states, target_sample_size, max_n_states):
    sample = []
    max_n_states = min(max_n_states, len(states))
    for state in reversed(states[-max_n_states:]):
        raw = history.get_fvals(state.acceptance_indices)
        demeaned = raw - raw.mean()
        sample += demeaned.tolist()
        if len(sample) >= target_sample_size:
            break

    sigma = np.var(sample, ddof=1)

    return sigma


def _unweighted_vector(history, states, target_sample_size, max_n_states):
    sample = []
    max_n_states = min(max_n_states, len(states))
    for state in reversed(states[-max_n_states:]):
        raw = history.get_fvecs(state.acceptance_indices)
        demeaned = raw - raw.mean(axis=0)
        sample += demeaned.tolist()
        if len(sample) >= target_sample_size:
            break

    sigma = np.cov(sample, rowvar=False, ddof=1)

    return sigma

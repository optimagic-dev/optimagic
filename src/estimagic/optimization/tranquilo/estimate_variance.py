"""Estimate the variance or covariance matrix of the noise in the objective function."""

import numpy as np

from estimagic.optimization.tranquilo.get_component import get_component


def get_variance_estimator(fitter, user_options):
    func_dict = {
        "unweighted": _estimate_variance_unweighted,
    }

    default_options = {
        "max_distance_factor": 2,
        "max_n_states": 10,
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
    history,
    states,
    model_type,
    acceptance_indices,
    max_n_states,
    max_distance_factor,
):
    center_indices = _get_admissible_center_indices(
        states=states,
        history=history,
        max_n_states=max_n_states,
        max_distance_factor=max_distance_factor,
    )

    sample = []
    for center_index in center_indices:
        indices = acceptance_indices[center_index]
        if model_type == "scalar":
            raw = history.get_fvals(indices)
            demeaned = raw - raw.mean()
        else:
            raw = history.get_fvecs(indices)
            demeaned = raw - raw.mean(axis=0)

        sample += demeaned.tolist()

    if model_type == "scalar":
        sigma = np.var(sample, ddof=1)
    else:
        sigma = np.cov(sample, rowvar=False, ddof=1)

    return sigma


def _get_admissible_center_indices(states, history, max_n_states, max_distance_factor):
    max_n_states = min(max_n_states, len(states))
    states = states[-max_n_states:]

    candidate_indices = {state.index for state in states} | {
        state.candidate_index for state in states
    }
    candidate_indices = np.array(sorted(candidate_indices))

    xs = history.get_xs(candidate_indices)

    order = None if states[-1].trustregion.shape == "sphere" else np.inf

    dists = np.linalg.norm(xs - states[-1].x, axis=1, ord=order)

    cutoff = max_distance_factor * states[-1].trustregion.radius

    mask = dists < cutoff

    admissible = list(candidate_indices[mask])

    return admissible

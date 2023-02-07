"""Estimate the variance or covariance matrix of the noise in the objective function."""

import numpy as np

from estimagic.optimization.tranquilo.get_component import get_component
from estimagic.optimization.tranquilo.options import Region


def get_variance_estimator(fitter, user_options, acceptance_options):
    func_dict = {
        "unweighted": _estimate_variance_unweighted,
    }

    default_options = {
        "max_n_states": 3,
        "acceptance_options": acceptance_options,
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
    history, states, model_type, max_n_states, acceptance_options
):
    max_n_states = min(max_n_states, len(states))
    if model_type == "scalar":
        out = _unweighted_scalar(history, states[-max_n_states:], acceptance_options)
    elif model_type == "vector":
        out = _unweighted_vector(history, states[-max_n_states:], acceptance_options)
    else:
        raise ValueError("model_type must be scalar or vector.")

    return out


def _unweighted_scalar(history, states, acceptance_options):
    regions = _get_search_regions(states=states, acceptance_options=acceptance_options)

    sample = []
    for region in regions:
        indices = history.get_indices_in_region(region)
        raw = history.get_fvals(indices)
        demeaned = raw - raw.mean()
        sample += demeaned.tolist()

    sigma = np.var(sample, ddof=1)

    return sigma


def _unweighted_vector(history, states, acceptance_options):
    regions = _get_search_regions(states=states, acceptance_options=acceptance_options)

    sample = []
    for region in regions:
        indices = history.get_indices_in_region(region)
        raw = history.get_fvecs(indices)
        demeaned = raw - raw.mean(axis=0)
        sample += demeaned.tolist()

    sigma = np.cov(sample, rowvar=False, ddof=1)

    return sigma


def _get_search_regions(states, acceptance_options):
    regions = {}

    for state in reversed(states):
        # ==============================================================================
        # from accepted indices
        # ==============================================================================
        radius = state.trustregion.radius * acceptance_options.radius_factor
        if state.index in regions:
            radius = max(radius, regions[state.index].radius)

        region = Region(
            center=state.x,
            radius=radius,
            shape="cube",
        )

        regions[state.index] = region

        # ==============================================================================
        # from rejected indices
        # ==============================================================================

        radius = state.trustregion.radius * acceptance_options.radius_factor
        if state.candidate_index in regions:
            radius = max(radius, regions[state.index].radius)

        region = Region(
            center=state.candidate_x,
            radius=radius,
            shape="cube",
        )

        regions[state.candidate_index] = region

    return list(regions.values())

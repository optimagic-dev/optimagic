"""Functions that decide what is the next accepted point, given a candidate.

Decision functions can simply decide whether or not the candidate is accepted but can
also do own function evaluations and decide to accept a different point.

"""
from typing import NamedTuple

import numpy as np

from estimagic.optimization.tranquilo.acceptance_sample_size import (
    get_acceptance_sample_sizes,
)
from estimagic.optimization.tranquilo.get_component import get_component
from estimagic.optimization.tranquilo.options import Region


def get_acceptance_decider(acceptance_decider, acceptance_options, sampler):
    func_dict = {
        "classic": accept_classic,
        "naive_noisy": accept_naive_noisy,
        "noisy": accept_noisy,
    }

    default_options = {
        "acceptance_options": acceptance_options,
        "sampler": sampler,
    }

    out = get_component(
        name_or_func=acceptance_decider,
        func_dict=func_dict,
        component_name="acceptance_decider",
        default_options=default_options,
    )

    return out


def accept_classic(
    subproblem_solution,
    state,
    acceptance_indices,
    *,
    wrapped_criterion,
    acceptance_options,
):
    """Do a classic acceptance step for a trustregion algorithm.

    Args:
        subproblem_solution (SubproblemResult): Result of the subproblem solution.
        state (State): Namedtuple containing the trustregion, criterion value of
            previously accepted point, indices of model points, etc.
        acceptance_indices (dict): TODO.
        wrapped_criterion (callable): The criterion function.
        acceptance_options (dict): Options for the acceptance step.

    Returns:
        AcceptanceResult

    """
    candidate_x = subproblem_solution.x
    _, candidate_fval, candidate_index = wrapped_criterion(candidate_x)
    actual_improvement = -(candidate_fval - state.fval)

    acceptance_indices[candidate_index] = [candidate_index]

    rho = calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    is_accepted = actual_improvement >= acceptance_options.min_improvement

    res = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        old_state=state,
    )

    return res, acceptance_indices


def accept_naive_noisy(
    subproblem_solution,
    state,
    rng,
    acceptance_indices,
    *,
    sampler,
    wrapped_criterion,
    acceptance_options,
):
    """Do a naive noisy acceptance step, averaging over a fixed number of points."""
    candidate_x = subproblem_solution.x
    acceptance_region = Region(
        center=candidate_x,
        radius=state.trustregion.radius * acceptance_options.radius_factor,
        shape=state.trustregion.shape,
    )
    sample = sampler(
        trustregion=acceptance_region,
        n_points=acceptance_options.n_initial,
        rng=rng,
    )

    xs = np.vstack([candidate_x, sample])

    _, _fvals, _indices = wrapped_criterion(xs)

    candidate_fval = np.mean(_fvals)
    candidate_index = _indices[0]

    acceptance_indices[candidate_index] = list(_indices)

    actual_improvement = -(candidate_fval - state.fval)

    rho = calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    is_accepted = actual_improvement >= acceptance_options.min_improvement

    res = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        old_state=state,
    )

    return res, acceptance_indices


def accept_noisy(
    subproblem_solution,
    state,
    rng,
    acceptance_indices,
    noise_variance,
    history,
    *,
    sampler,
    wrapped_criterion,
    acceptance_options,
):
    # ==================================================================================
    # Get additional sample sizes at currently accepted and candidate point
    # ==================================================================================
    n_1, n_2 = get_acceptance_sample_sizes(
        sigma=np.sqrt(noise_variance),
        existing_n1=len(acceptance_indices[state.index]),
        expected_improvement=subproblem_solution.expected_improvement,
        acceptance_options=acceptance_options,
    )

    acceptance_radius = state.trustregion.radius * acceptance_options.radius_factor

    # ==================================================================================
    # Sample at currently accepted point if necessary
    # ==================================================================================
    if n_1 > 0:
        sample_1 = sampler(
            trustregion=state.trustregion._replace(radius=acceptance_radius),
            n_points=n_1,
            rng=rng,
        )
        _, _, _indices_1 = wrapped_criterion(sample_1)

        acceptance_indices[state.index] += list(_indices_1)

    # ==================================================================================
    # Sample at candidate point
    # ==================================================================================
    candidate_x = subproblem_solution.x
    sample_2 = sampler(
        trustregion=state.trustregion._replace(
            center=candidate_x, radius=acceptance_radius
        ),
        n_points=n_2 - 1,
        rng=rng,
    )

    xs = np.vstack([candidate_x, sample_2])

    _, _, _indices_2 = wrapped_criterion(xs)

    candidate_index = _indices_2[0]

    acceptance_indices[candidate_index] = list(_indices_2)

    # ==================================================================================
    # Actual acceptance decision
    # ==================================================================================

    current_fval = history.get_fvals(acceptance_indices[state.index]).mean()
    candidate_fval = history.get_fvals(acceptance_indices[candidate_index]).mean()

    actual_improvement = -(candidate_fval - current_fval)

    rho = calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    is_accepted = actual_improvement >= acceptance_options.min_improvement

    res = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        old_state=state,
    )

    return res, acceptance_indices


class AcceptanceResult(NamedTuple):
    x: np.ndarray
    fval: float
    index: int
    rho: float
    accepted: bool
    step_length: float
    relative_step_length: float
    candidate_index: int
    candidate_x: np.ndarray


def _get_acceptance_result(
    candidate_x,
    candidate_fval,
    candidate_index,
    rho,
    is_accepted,
    old_state,
):
    x = candidate_x if is_accepted else old_state.x
    fval = candidate_fval if is_accepted else old_state.fval
    index = candidate_index if is_accepted else old_state.index
    step_length = np.linalg.norm(x - old_state.x, ord=2)
    relative_step_length = step_length / old_state.trustregion.radius

    out = AcceptanceResult(
        x=x,
        fval=fval,
        index=index,
        rho=rho,
        accepted=is_accepted,
        step_length=step_length,
        relative_step_length=relative_step_length,
        candidate_index=candidate_index,
        candidate_x=candidate_x,
    )
    return out


def calculate_rho(actual_improvement, expected_improvement):
    if expected_improvement == 0 and actual_improvement > 0:
        rho = np.inf
    elif expected_improvement == 0:
        rho = -np.inf
    else:
        rho = actual_improvement / expected_improvement
    return rho

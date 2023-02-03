"""Functions that decide what is the next accepted point, given a candidate.

Decision functions can simply decide whether or not the candidate is accepted but can
also do own function evaluations and decide to accept a different point.

"""
from typing import NamedTuple

import numpy as np

from estimagic.optimization.tranquilo.get_component import get_component
from estimagic.optimization.tranquilo.options import TrustRegion


def get_acceptance_decider(acceptance_decider, acceptance_options, sampler):
    func_dict = {
        "classic": accept_classic,
        "naive_noisy": accept_naive_noisy,
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
    *,
    wrapped_criterion,
    acceptance_options,
):
    """Do a classic acceptance step for a trustregion algorithm.

    Args:
        subproblem_solution (SubproblemResult): Result of the subproblem solution.
        state (State): Namedtuple containing the trustregion, criterion value of
            previously accepted point, indices of model points, etc.
        criterion (callable): The criterion function.
        acceptance_options (dict): Options for the acceptance step.
        batch_size (int): The batch size.

    Returns:
        AcceptanceResult

    """
    candidate_x = subproblem_solution.x
    _, candidate_fval, candidate_index = wrapped_criterion(candidate_x)
    actual_improvement = -(candidate_fval - state.fval)

    rho = _calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    is_accepted = actual_improvement >= acceptance_options.min_improvement

    out = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        acceptance_indices=np.array([candidate_index]),
        old_state=state,
    )

    return out


def accept_naive_noisy(
    subproblem_solution,
    state,
    rng,
    *,
    sampler,
    wrapped_criterion,
    acceptance_options,
):
    """Do a naive noisy acceptance step, averaging over a fixed number of points."""
    candidate_x = subproblem_solution.x
    acceptance_region = TrustRegion(
        center=candidate_x,
        radius=state.trustregion.radius * acceptance_options.radius_factor,
    )
    sample = sampler(
        trustregion=acceptance_region,
        n_points=acceptance_options.n_initial,
        rng=rng,
    )

    xs = np.vstack([candidate_x, sample])

    _, acceptance_fvals, acceptance_indices = wrapped_criterion(xs)

    candidate_fval = np.mean(acceptance_fvals)
    candidate_index = acceptance_indices[0]

    actual_improvement = -(candidate_fval - state.fval)

    rho = _calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    is_accepted = actual_improvement >= acceptance_options.min_improvement

    out = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        acceptance_indices=acceptance_indices,
        old_state=state,
    )

    return out


class AcceptanceResult(NamedTuple):
    x: np.ndarray
    fval: float
    index: int
    rho: float
    accepted: bool
    step_length: float
    relative_step_length: float
    acceptance_indices: np.ndarray


def _get_acceptance_result(
    candidate_x,
    candidate_fval,
    candidate_index,
    rho,
    is_accepted,
    acceptance_indices,
    old_state,
):
    x = candidate_x if is_accepted else old_state.x
    fval = candidate_fval if is_accepted else old_state.fval
    step_length = np.linalg.norm(x - old_state.x)
    relative_step_length = step_length / old_state.trustregion.radius

    out = AcceptanceResult(
        x=x,
        fval=fval,
        index=candidate_index if is_accepted else old_state.index,
        rho=rho,
        accepted=is_accepted,
        step_length=step_length,
        relative_step_length=relative_step_length,
        acceptance_indices=acceptance_indices,
    )
    return out


def _calculate_rho(actual_improvement, expected_improvement):
    if expected_improvement == 0 and actual_improvement > 0:
        rho = np.inf
    elif expected_improvement == 0:
        rho = -np.inf
    else:
        rho = actual_improvement / expected_improvement
    return rho

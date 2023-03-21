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
from estimagic.optimization.tranquilo.options import AcceptanceOptions


def get_acceptance_decider(acceptance_decider, acceptance_options):
    func_dict = {
        "classic": _accept_classic,
        "naive_noisy": accept_naive_noisy,
        "noisy": accept_noisy,
    }

    out = get_component(
        name_or_func=acceptance_decider,
        func_dict=func_dict,
        component_name="acceptance_decider",
        user_options=acceptance_options,
        default_options=AcceptanceOptions(),
    )

    return out


def _accept_classic(
    subproblem_solution,
    state,
    history,
    *,
    wrapped_criterion,
    min_improvement,
):
    """Do a classic acceptance step for a trustregion algorithm.

    Args:
        subproblem_solution (SubproblemResult): Result of the subproblem solution.
        state (State): Namedtuple containing the trustregion, criterion value of
            previously accepted point, indices of model points, etc.
        wrapped_criterion (callable): The criterion function.
        min_improvement (float): Minimum improvement required to accept a point.

    Returns:
        AcceptanceResult

    """
    out = _accept_simple(
        subproblem_solution=subproblem_solution,
        state=state,
        history=history,
        wrapped_criterion=wrapped_criterion,
        min_improvement=min_improvement,
        n_evals=1,
    )
    return out


def accept_naive_noisy(
    subproblem_solution,
    state,
    history,
    *,
    wrapped_criterion,
    min_improvement,
):
    """Do a naive noisy acceptance step, averaging over a fixed number of points."""
    out = _accept_simple(
        subproblem_solution=subproblem_solution,
        state=state,
        history=history,
        wrapped_criterion=wrapped_criterion,
        min_improvement=min_improvement,
        n_evals=5,
    )
    return out


def _accept_simple(
    subproblem_solution,
    state,
    history,
    *,
    wrapped_criterion,
    min_improvement,
    n_evals,
):
    """Do a classic acceptance step for a trustregion algorithm.

    Args:
        subproblem_solution (SubproblemResult): Result of the subproblem solution.
        state (State): Namedtuple containing the trustregion, criterion value of
            previously accepted point, indices of model points, etc.
        wrapped_criterion (callable): The criterion function.
        min_improvement (float): Minimum improvement required to accept a point.

    Returns:
        AcceptanceResult

    """
    candidate_x = subproblem_solution.x

    candidate_index = history.add_xs(candidate_x)

    wrapped_criterion({candidate_index: n_evals})

    candidate_fval = np.mean(history.get_fvals(candidate_index))

    actual_improvement = -(candidate_fval - state.fval)

    rho = calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    is_accepted = actual_improvement >= min_improvement

    res = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        old_state=state,
    )

    return res


def accept_noisy(
    subproblem_solution,
    state,
    noise_variance,
    history,
    *,
    wrapped_criterion,
    min_improvement,
    power_level,
    confidence_level,
    n_min,
    n_max,
):
    candidate_x = subproblem_solution.x
    candidate_index = history.add_xs(candidate_x)
    existing_n1 = len(history.get_fvals(state.index))

    n_1, n_2 = get_acceptance_sample_sizes(
        sigma=np.sqrt(noise_variance),
        existing_n1=existing_n1,
        expected_improvement=subproblem_solution.expected_improvement,
        power_level=power_level,
        confidence_level=confidence_level,
        n_min=n_min,
        n_max=n_max,
    )

    eval_info = {
        state.index: n_1,
        candidate_index: n_2,
    }

    wrapped_criterion(eval_info)

    current_fval = history.get_fvals(state.index).mean()
    candidate_fval = history.get_fvals(candidate_index).mean()

    actual_improvement = -(candidate_fval - current_fval)

    rho = calculate_rho(
        actual_improvement=actual_improvement,
        expected_improvement=subproblem_solution.expected_improvement,
    )

    is_accepted = actual_improvement >= min_improvement

    res = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        is_accepted=is_accepted,
        old_state=state,
    )

    return res


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

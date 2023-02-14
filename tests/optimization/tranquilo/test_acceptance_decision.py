from collections import namedtuple
from functools import partial

import numpy as np
import pytest
from estimagic.optimization.tranquilo.acceptance_decision import (
    _get_acceptance_result,
    accept_classic,
    accept_naive_noisy,
    calculate_rho,
)
from estimagic.optimization.tranquilo.options import (
    AcceptanceOptions,
    Bounds,
)
from estimagic.optimization.tranquilo.sample_points import get_sampler
from estimagic.optimization.tranquilo.solve_subproblem import SubproblemResult
from estimagic.optimization.tranquilo.tranquilo import Region
from numpy.testing import assert_array_equal

# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture()
def subproblem_solution():
    res = SubproblemResult(
        x=1 + np.arange(2.0),
        expected_improvement=1.0,
        n_iterations=1,
        success=True,
        centered_x=None,
    )
    return res


@pytest.fixture()
def acceptance_indices():
    return {0: [0]}


@pytest.fixture()
def acceptance_options():
    return AcceptanceOptions()


@pytest.fixture()
def wrapped_criterion():
    def _wrapped_criterion(x, stochastic=False):
        rng = np.random.default_rng(0)
        out = (None, 0.5, 2)  # (_, candidate_fval, candidate_index)
        if x.ndim > 1:
            candidate_fval = []
            for _ in x:
                noise = rng.normal(scale=0.01) if stochastic else 0
                candidate_fval.append(out[1] + noise)
            candidate_index = len(x) * [out[2]]
            out = (None, candidate_fval, candidate_index)
        return out

    return _wrapped_criterion


@pytest.fixture()
def sampler():
    return get_sampler("sphere", Bounds(None, None))


# ======================================================================================
# Test accept_xxx
# ======================================================================================


trustregion = Region(center=np.zeros(2), radius=2.0)
State = namedtuple("State", "x trustregion fval index")
states = [  # we will parametrize over `states`
    State(np.arange(2.0), trustregion, 0.25, 0),  # better than candidate
    State(np.arange(2.0), trustregion, 1, 0),  # worse than candidate
]


@pytest.mark.parametrize("state", states)
def test_accept_classic(
    state,
    subproblem_solution,
    acceptance_indices,
    wrapped_criterion,
    acceptance_options,
):
    res_got, acceptance_indices_got = accept_classic(
        subproblem_solution=subproblem_solution,
        state=state,
        acceptance_indices=acceptance_indices,
        wrapped_criterion=wrapped_criterion,
        acceptance_options=acceptance_options,
    )

    accept_candidate = state.fval > 0.5

    assert acceptance_indices_got == {0: [0], 2: [2]}
    assert res_got.accepted == accept_candidate
    assert res_got.rho == -(0.5 - state.fval)
    assert res_got.fval == (0.5 if accept_candidate else state.fval)
    assert res_got.index == (2 if accept_candidate else state.index)
    assert res_got.candidate_index == 2
    assert_array_equal(
        res_got.x, subproblem_solution.x if accept_candidate else state.x
    )
    assert_array_equal(res_got.candidate_x, 1.0 + np.arange(2))


@pytest.mark.parametrize("state", states)
def test_accept_naive_noisy_deterministic(
    state,
    subproblem_solution,
    acceptance_indices,
    wrapped_criterion,
    acceptance_options,
    sampler,
):
    res_got, acceptance_indices_got = accept_naive_noisy(
        subproblem_solution=subproblem_solution,
        state=state,
        rng=np.random.default_rng(0),
        acceptance_indices=acceptance_indices,
        sampler=sampler,
        wrapped_criterion=wrapped_criterion,
        acceptance_options=acceptance_options,
    )

    accept_candidate = state.fval > 0.5

    assert acceptance_indices_got == {
        0: [0],
        2: (acceptance_options.n_initial + 1) * [2],
    }
    assert res_got.accepted == accept_candidate
    assert res_got.rho == -(0.5 - state.fval)
    assert res_got.fval == (0.5 if accept_candidate else state.fval)
    assert res_got.index == (2 if accept_candidate else state.index)
    assert res_got.candidate_index == 2
    assert_array_equal(
        res_got.x, subproblem_solution.x if accept_candidate else state.x
    )
    assert_array_equal(res_got.candidate_x, 1.0 + np.arange(2))


@pytest.mark.parametrize("state", states)
def test_accept_naive_noisy_stochastic(
    state,
    subproblem_solution,
    acceptance_indices,
    wrapped_criterion,
    acceptance_options,
    sampler,
):
    res_got, acceptance_indices_got = accept_naive_noisy(
        subproblem_solution=subproblem_solution,
        state=state,
        rng=np.random.default_rng(0),
        acceptance_indices=acceptance_indices,
        sampler=sampler,
        wrapped_criterion=partial(wrapped_criterion, stochastic=True),
        acceptance_options=acceptance_options,
    )

    accept_candidate = state.fval > 0.5

    assert acceptance_indices_got == {
        0: [0],
        2: (acceptance_options.n_initial + 1) * [2],
    }
    assert res_got.accepted == accept_candidate
    # test relative difference because of stochastic criterion
    assert abs(res_got.rho + (0.5 - state.fval)) < 0.01
    assert abs(res_got.fval - (0.5 if accept_candidate else state.fval)) < 0.01
    assert res_got.index == (2 if accept_candidate else state.index)
    assert res_got.candidate_index == 2
    assert_array_equal(
        res_got.x, subproblem_solution.x if accept_candidate else state.x
    )
    assert_array_equal(res_got.candidate_x, 1.0 + np.arange(2))


@pytest.mark.xfail(reason="not implemented yet")
def test_accept_noisy():
    pass


# ======================================================================================
# Test _get_acceptance_result
# ======================================================================================


def test_get_acceptance_result():
    candidate_x = 1 + np.arange(2)
    candidate_fval = 0
    candidate_index = 0
    rho = 1
    tr = Region(center=np.zeros(2), radius=2.0)
    old_state = namedtuple("State", "x fval index trustregion")(np.arange(2), 1, 1, tr)

    ar_when_accepted = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        old_state=old_state,
        is_accepted=True,
    )

    assert_array_equal(ar_when_accepted.x, candidate_x)
    assert ar_when_accepted.fval == candidate_fval
    assert ar_when_accepted.index == candidate_index
    assert ar_when_accepted.accepted is True
    assert ar_when_accepted.step_length == np.sqrt(2)
    assert ar_when_accepted.relative_step_length == np.sqrt(2) / 2

    ar_when_not_accepted = _get_acceptance_result(
        candidate_x=candidate_x,
        candidate_fval=candidate_fval,
        candidate_index=candidate_index,
        rho=rho,
        old_state=old_state,
        is_accepted=False,
    )

    assert_array_equal(ar_when_not_accepted.x, old_state.x)
    assert ar_when_not_accepted.fval == old_state.fval
    assert ar_when_not_accepted.index == old_state.index
    assert ar_when_not_accepted.accepted is False
    assert ar_when_not_accepted.step_length == 0
    assert ar_when_not_accepted.relative_step_length == 0


# ======================================================================================
# Test calculate_rho
# ======================================================================================


CASES = [
    (0, 0, -np.inf),
    (-1, 0, -np.inf),
    (1, 0, np.inf),
    (0, 1, 0),
    (1, 2, 1 / 2),
]


@pytest.mark.parametrize("actual_improvement, expected_improvement, expected", CASES)
def test_calculate_rho(actual_improvement, expected_improvement, expected):
    rho = calculate_rho(actual_improvement, expected_improvement)
    assert rho == expected

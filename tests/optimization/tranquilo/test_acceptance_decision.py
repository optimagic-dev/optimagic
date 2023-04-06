from collections import namedtuple

import numpy as np
import pytest
from tranquilo.optimization.tranquilo.acceptance_decision import (
    _accept_simple,
    _get_acceptance_result,
    calculate_rho,
)
from tranquilo.optimization.tranquilo.history import History
from tranquilo.optimization.tranquilo.region import Region
from tranquilo.optimization.tranquilo.solve_subproblem import SubproblemResult
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
        x_unit=None,
        shape=None,
    )
    return res


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
def test_accept_simple(
    state,
    subproblem_solution,
):
    history = History(functype="scalar")

    idxs = history.add_xs(np.arange(10).reshape(5, 2))

    history.add_evals(idxs.repeat(2), np.arange(10))

    def wrapped_criterion(eval_info):
        indices = np.array(list(eval_info)).repeat(np.array(list(eval_info.values())))
        history.add_evals(indices, -indices)

    res_got = _accept_simple(
        subproblem_solution=subproblem_solution,
        state=state,
        history=history,
        wrapped_criterion=wrapped_criterion,
        min_improvement=0.0,
        n_evals=2,
    )

    assert res_got.accepted
    assert res_got.index == 5
    assert res_got.candidate_index == 5
    assert_array_equal(res_got.x, subproblem_solution.x)
    assert_array_equal(res_got.candidate_x, 1.0 + np.arange(2))


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

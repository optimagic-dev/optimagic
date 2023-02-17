from collections import namedtuple

import numpy as np
import pytest
from estimagic.optimization.tranquilo.estimate_variance import (
    _estimate_variance_unweighted,
    _get_admissible_center_indices,
)
from estimagic.optimization.tranquilo.tranquilo import Region
from estimagic.optimization.tranquilo.tranquilo_history import History
from numpy.testing import assert_equal

# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture()
def criterion():
    return lambda x: (-(x**2), -np.sum(x**2))


@pytest.fixture()
def states_and_histories(criterion):
    """Return states and histories.

    Creates scenario of two states, each with three samples. Histories are created for
    the scalar and least_squares criterion.

    """
    State = namedtuple("State", "x trustregion index candidate_index")

    # Iteration 1
    # ==================================================================================

    # trustregion
    tr0 = Region(center=np.zeros(2), radius=np.sqrt(2))
    # samples
    x0 = np.array([[-1, 1], [0, 0], [1, 1]])
    # state
    state0 = State(x=np.ones(2), trustregion=tr0, index=2, candidate_index=2)

    # Iteration 2
    # ==================================================================================

    tr1 = Region(center=np.ones(2), radius=1)
    x1 = np.array([[0, 1], [1, 0], [2, 1]])
    state1 = State(x=np.array([2, 1]), trustregion=tr1, index=5, candidate_index=5)

    # History
    # ==================================================================================
    histories = {
        functype: History(functype=functype) for functype in ("scalar", "least_squares")
    }

    for _x in np.vstack((x0, x1)):
        fvec, fval = criterion(_x)
        histories["scalar"].add_entries(_x, fval)
        histories["least_squares"].add_entries(_x, fvec)

    return [state0, state1], histories


# ======================================================================================
# Test _get_admissible_center_indices
# ======================================================================================


TEST_CASES = [
    (1, 0, []),
    (1, 1e9, [5]),
    (2, 1 + 1e9, [2, 5]),
    (2, 1, [5]),
    (2, 0, []),
    (3, 1 + 1e9, [2, 5]),
    (3, 1, [5]),
    (3, 0, []),
]


@pytest.mark.parametrize("max_n_states, max_distance_factor, expected", TEST_CASES)
@pytest.mark.parametrize("functype", ["scalar", "least_squares"])
def test_get_admissible_center_indices(
    max_n_states, max_distance_factor, expected, functype, states_and_histories
):
    states, histories = states_and_histories
    history = histories[functype]
    admissible = _get_admissible_center_indices(
        states=states,
        history=history,
        max_n_states=max_n_states,
        max_distance_factor=max_distance_factor,
    )
    assert admissible == expected


# ======================================================================================
# Test _estimate_variance_unweighted
# ======================================================================================


TEST_CASES = [
    (1, 0, "scalar", np.nan),
    (1, 1e9, "scalar", np.var([-2, 2], ddof=1)),
    (2, 1 + 1e9, "scalar", np.var([-1, 1, -2, 2], ddof=1)),
    (2, 1, "scalar", np.var([-2, 2], ddof=1)),
    (2, 0, "scalar", np.nan),
    (1, 0, "least_squares", np.nan),
    (
        1,
        1e9,
        "least_squares",
        np.cov([[3 / 2, 1 / 2], [-3 / 2, -1 / 2]], rowvar=False, ddof=1),
    ),
    (
        2,
        1 + 1e9,
        "least_squares",
        np.cov(
            [[3 / 2, 1 / 2], [-3 / 2, -1 / 2], [1 / 2, 1 / 2], [-1 / 2, -1 / 2]],
            rowvar=False,
            ddof=1,
        ),
    ),
    (
        2,
        1,
        "least_squares",
        np.cov([[3 / 2, 1 / 2], [-3 / 2, -1 / 2]], rowvar=False, ddof=1),
    ),
    (2, 0, "least_squares", np.nan),
]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # caused by undefined variances
@pytest.mark.parametrize(
    "max_n_states, max_distance_factor, functype, expected", TEST_CASES
)
def test_estimate_variance_unweighted(
    max_n_states,
    max_distance_factor,
    functype,
    expected,
    states_and_histories,
):
    states, histories = states_and_histories
    history = histories[functype]

    acceptance_indices = {
        2: [1, 2],
        5: [4, 5],
    }

    estimate = _estimate_variance_unweighted(
        history=history,
        states=states,
        model_type=functype,
        acceptance_indices=acceptance_indices,
        max_n_states=max_n_states,
        max_distance_factor=max_distance_factor,
    )

    assert_equal(expected, estimate)

from collections import namedtuple

import numpy as np
import pytest
from estimagic.optimization.tranquilo.estimate_variance import (
    _estimate_variance_unweighted,
    _get_admissible_center_indices,
)
from estimagic.optimization.tranquilo.tranquilo import Region
from estimagic.optimization.tranquilo.tranquilo_history import History

# ======================================================================================
# Fixtures
# ======================================================================================


@pytest.fixture()
def criterion():
    return lambda x: (x**2, np.sum(x**2))


@pytest.fixture()
def states_and_histories(criterion):
    """Create list of two states and two histories with scalar and least_squares
    func."""
    # ==================================================================================
    # States
    # ==================================================================================
    tr0 = Region(center=np.zeros(2), radius=0.5)
    tr1 = Region(center=np.ones(2), radius=0.5)

    State = namedtuple("State", "x trustregion index candidate_index")

    states = [
        State(x=np.zeros(2), trustregion=tr0, index=1, candidate_index=2),
        State(x=np.ones(2), trustregion=tr1, index=5, candidate_index=5),
    ]
    # ==================================================================================
    # Histories
    # ==================================================================================
    x0 = np.array([[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]])
    x1 = np.array([[0.4, 0.4], [1.0, 1.0], [1.6, 1.6]])

    _history_scalar = History(functype="scalar")
    _history_least_squares = History(functype="least_squares")
    histories = {"scalar": _history_scalar, "least_squares": _history_least_squares}
    for x in np.vstack((x0, x1)):
        fvec, fval = criterion(x)
        histories["scalar"].add_entries(x, fval)
        histories["least_squares"].add_entries(x, fvec)

    return states, histories


# ======================================================================================
# Test _get_admissible_center_indices
# ======================================================================================


TEST_CASES = [
    (1, 2, [5]),
    (2, 2.5, [2, 5]),
    (2, 3, [1, 2, 5]),
    (2, 2.5, [2, 5]),
    (2, 0, []),
]


@pytest.mark.parametrize("max_n_states, max_distance_factor, expected", TEST_CASES)
@pytest.mark.parametrize("functype", ["scalar", "least_squares"])
def test_get_admissible_center_indices(
    max_n_states, max_distance_factor, expected, functype, states_and_histories
):
    states, histories = states_and_histories
    history = histories[functype]
    admissible = _get_admissible_center_indices(
        states,
        history,
        max_n_states=max_n_states,
        max_distance_factor=max_distance_factor,
    )
    assert admissible == expected


# ======================================================================================
# Test _estimate_variance_unweighted
# ======================================================================================


TEST_CASES = [
    (1, 2, "scalar", np.nan),
    (2, 2.5, "scalar", 0.0081),
    (2, 3, "scalar", 0.0353),
    (2, 2.5, "scalar", 0.0081),
    (2, 0, "scalar", np.nan),
    (2, 0, "least_squares", np.nan),
]


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
        1: [0, 1],
        2: [2, 3],
        5: [5],
    }

    estimate = _estimate_variance_unweighted(
        history=history,
        states=states,
        model_type=functype,
        acceptance_indices=acceptance_indices,
        max_n_states=max_n_states,
        max_distance_factor=max_distance_factor,
    )

    if np.isnan(expected):
        assert np.isnan(estimate)
    else:
        assert abs(expected - estimate) < 1e-10


TEST_CASES = [
    (1, 2, "least_squares", np.nan),
    (2, 2.5, "least_squares", 0.0081),
    (2, 3, "scalar", 0.0353),
    (2, 2.5, "scalar", 0.0081),
]


@pytest.mark.xfail(reason="least squares part not implemented yet.")
@pytest.mark.parametrize(
    "max_n_states, max_distance_factor, functype, expected", TEST_CASES
)
def test_estimate_variance_unweighted_xfail(
    max_n_states,
    max_distance_factor,
    functype,
    expected,
    states_and_histories,
):
    states, histories = states_and_histories
    history = histories[functype]

    acceptance_indices = {
        1: [0, 1],
        2: [2, 3],
        5: [5],
    }

    estimate = _estimate_variance_unweighted(
        history=history,
        states=states,
        model_type=functype,
        acceptance_indices=acceptance_indices,
        max_n_states=max_n_states,
        max_distance_factor=max_distance_factor,
    )

    if np.isnan(expected):
        assert np.isnan(estimate)
    else:
        assert abs(expected - estimate) < 1e-10

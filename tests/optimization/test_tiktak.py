from itertools import product

import numpy as np
import pandas as pd
import pytest
from estimagic.optimization.optimize import process_multistart_sample
from estimagic.optimization.tiktak import _linear_weights
from estimagic.optimization.tiktak import _tiktak_weights
from estimagic.optimization.tiktak import draw_exploration_sample
from estimagic.optimization.tiktak import get_batched_optimization_sample
from estimagic.optimization.tiktak import get_internal_sampling_bounds
from estimagic.optimization.tiktak import run_explorations
from estimagic.optimization.tiktak import update_convergence_state
from numpy.testing import assert_array_almost_equal as aaae


@pytest.fixture
def params():
    df = pd.DataFrame(index=["a", "b", "c"])
    df["value"] = [0, 1, 2.0]
    df["soft_lower_bound"] = [-1, 0, np.nan]
    df["upper_bound"] = [2, 2, np.nan]
    return df


@pytest.fixture
def constraints():
    return [{"type": "fixed", "loc": "c", "value": 2}]


samples = [pd.DataFrame(np.ones((2, 3)), columns=["a", "b", "c"]), np.ones((2, 3))]


@pytest.mark.parametrize("sample", samples)
def test_process_multistart_sample(sample, params):

    calculated = process_multistart_sample(sample, params, lambda x: x)
    expeceted = np.ones((2, 3))
    aaae(calculated, expeceted)


distributions = ["triangle", "uniform"]
rules = [
    "random",
    "sobol",
    "halton",
    "hammersley",
    "korobov",
    "latin_hypercube",
    # chebyshev generated samples of the wrong size!
]
test_cases = list(product(distributions, rules))


@pytest.mark.parametrize("dist, rule", test_cases)
def test_draw_exploration_sample(dist, rule):

    results = []
    for _ in range(2):
        results.append(
            draw_exploration_sample(
                x=np.array([0.5, 0.5]),
                lower=np.zeros(2),
                upper=np.ones(2),
                n_samples=3,
                sampling_distribution=dist,
                sampling_method=rule,
                seed=1234,
            )
        )

    aaae(results[0], results[1])
    calculated = results[0]
    assert calculated.shape == (3, 2)


def test_get_internal_sampling_bounds(params, constraints):
    calculated = get_internal_sampling_bounds(params, constraints)
    expeceted = [np.array([-1, 0]), np.array([2, 2])]
    for calc, exp in zip(calculated, expeceted):
        aaae(calc, exp)


def test_run_explorations():
    def _dummy(x, **kwargs):
        assert set(kwargs) == {
            "task",
            "algorithm_info",
            "error_handling",
            "error_penalty",
            "fixed_log_data",
        }
        if x.sum() == 5:
            out = {"value": np.nan}
        else:
            out = {"value": -x.sum()}
        return out

    calculated = run_explorations(
        func=_dummy,
        sample=np.arange(6).reshape(3, 2),
        batch_evaluator="joblib",
        n_cores=1,
        step_id=0,
        error_handling="raise",
    )

    exp_values = np.array([-9, -1])
    exp_sample = np.array([[4, 5], [0, 1]])

    aaae(calculated["sorted_values"], exp_values)
    aaae(calculated["sorted_sample"], exp_sample)


def test_get_batched_optimization_sample():
    calculated = get_batched_optimization_sample(
        sorted_sample=np.arange(12).reshape(6, 2),
        n_optimizations=5,
        batch_size=4,
    )
    expected = [[[0, 1], [2, 3], [4, 5], [6, 7]], [[8, 9]]]

    assert len(calculated[0]) == 4
    assert len(calculated[1]) == 1
    assert len(calculated) == 2

    for calc_batch, exp_batch in zip(calculated, expected):
        assert isinstance(calc_batch, list)
        for calc_entry, exp_entry in zip(calc_batch, exp_batch):
            assert isinstance(calc_entry, np.ndarray)
            assert calc_entry.tolist() == exp_entry


def test_linear_weights():
    calculated = _linear_weights(5, 10, 0.4, 0.8)
    expected = 0.6
    assert np.allclose(calculated, expected)


def test_tiktak_weights():
    assert np.allclose(0.3, _tiktak_weights(0, 10, 0.3, 0.8))
    assert np.allclose(0.8, _tiktak_weights(10, 10, 0.3, 0.8))


@pytest.fixture
def current_state():
    state = {
        "best_x": np.ones(3),
        "best_y": 5,
        "best_res": None,
        "x_history": [np.arange(3) - 1e-20, np.ones(3)],
        "y_history": [6, 5],
        "result_history": [],
        "start_history": [],
    }

    return state


@pytest.fixture
def starts():
    return [np.zeros(3)]


@pytest.fixture
def results():
    return [{"solution_x": np.arange(3) + 1e-10, "solution_criterion": 4}]


def test_update_state_converged(current_state, starts, results):
    criteria = {
        "xtol": 1e-3,
        "max_discoveries": 2,
    }

    new_state, is_converged = update_convergence_state(
        current_state=current_state,
        starts=starts,
        results=results,
        convergence_criteria=criteria,
    )

    aaae(new_state["best_x"], np.arange(3))
    assert new_state["best_y"] == 4
    assert new_state["y_history"] == [6, 5, 4]
    assert new_state["result_history"][0]["solution_criterion"] == 4
    aaae(new_state["start_history"][0], np.zeros(3))
    assert new_state["best_res"].keys() == results[0].keys()

    assert is_converged


def test_update_state_not_converged(current_state, starts, results):
    criteria = {
        "xtol": 1e-3,
        "max_discoveries": 5,
    }

    _, is_converged = update_convergence_state(
        current_state=current_state,
        starts=starts,
        results=results,
        convergence_criteria=criteria,
    )

    assert not is_converged

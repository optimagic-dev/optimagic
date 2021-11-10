from itertools import product

import numpy as np
import pandas as pd
import pytest
from estimagic.optimization.tiktak import _do_actual_sampling
from estimagic.optimization.tiktak import _get_internal_sampling_bounds
from estimagic.optimization.tiktak import _has_transforming_constraints
from estimagic.optimization.tiktak import _linear_weights
from estimagic.optimization.tiktak import _process_sample
from estimagic.optimization.tiktak import _tiktak_weights
from estimagic.optimization.tiktak import get_batched_optimization_sample
from estimagic.optimization.tiktak import get_exploration_sample
from estimagic.optimization.tiktak import run_explorations
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


def test_get_exploration_sample_runs(params, constraints):
    calculated = get_exploration_sample(
        params,
        n_samples=30,
        sampling_distribution="uniform",
        sampling_method="sobol",
        constraints=constraints,
        seed=1234,
    )
    assert calculated.shape == (30, 2)


samples = [pd.DataFrame(np.ones((2, 3)), columns=["a", "b", "c"]), np.ones((2, 3))]


@pytest.mark.parametrize("sample", samples)
def test_process_sample(sample, params):
    calculated = _process_sample(sample, params, [])
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
def test_do_actual_sampling(dist, rule):

    results = []
    for _ in range(2):
        results.append(
            _do_actual_sampling(
                midpoint=np.array([0.5, 0.5]),
                lower=np.zeros(2),
                upper=np.ones(2),
                size=3,
                distribution=dist,
                rule=rule,
                seed=1234,
            )
        )

    aaae(results[0], results[1])
    calculated = results[0]
    assert calculated.shape == (3, 2)


def test_get_internal_sampling_bounds(params, constraints):
    calculated = _get_internal_sampling_bounds(params, constraints)
    expeceted = [np.array([-1, 0]), np.array([2, 2])]
    for calc, exp in zip(calculated, expeceted):
        aaae(calc, exp)


def test_has_transforming_constraints():
    constraints = [{"type": "sdcorr"}]
    assert _has_transforming_constraints(constraints)

    constraints = [{"type": "fixed"}]
    assert not _has_transforming_constraints(constraints)


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
            out = np.nan
        else:
            out = -x.sum()
        return out

    calculated = run_explorations(
        func=_dummy,
        sample=np.arange(6).reshape(3, 2),
        batch_evaluator="joblib",
        n_cores=1,
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

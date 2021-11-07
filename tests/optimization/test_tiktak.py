from itertools import product

import numpy as np
import pandas as pd
import pytest
from estimagic.optimization.tiktak import _do_actual_sampling
from estimagic.optimization.tiktak import _get_internal_sampling_bounds
from estimagic.optimization.tiktak import _has_transforming_constraints
from estimagic.optimization.tiktak import _process_sample
from estimagic.optimization.tiktak import get_exploration_sample
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
    calculated = get_exploration_sample(params, constraints=constraints)
    assert calculated.shape == (30, 3)


samples = [pd.DataFrame(np.ones((2, 3)), columns=["a", "b", "c"]), np.ones((2, 3))]


@pytest.mark.parametrize("sample", samples)
def test_process_sample(sample, params):
    calculated = _process_sample(sample, params)
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

"""Test optimization helper functions."""

import warnings

import numpy as np
import pytest

from optimagic.algorithms import ALL_ALGORITHMS
from optimagic.config import IS_PYGMO_INSTALLED
from optimagic.optimization.optimize import minimize
from optimagic.optimizers.pygmo_optimizers import (
    _convert_str_to_int,
    get_population_size,
)
from optimagic.parameters.bounds import Bounds

pytestmark = pytest.mark.pygmo

test_cases = [
    # popsize, x, lower_bound, expected
    (55.3, None, None, 55),
    (None, np.ones(5), 500, 500),
    (None, np.ones(5), 4, 60),
]


@pytest.mark.parametrize("popsize, x, lower_bound, expected", test_cases)
def test_determine_population_size(popsize, x, lower_bound, expected):
    res = get_population_size(population_size=popsize, x=x, lower_bound=lower_bound)
    assert res == expected


def test_convert_str_to_int():
    d = {"a": 1, "b": 3}
    assert _convert_str_to_int(d, "a") == 1
    assert _convert_str_to_int(d, 1) == 1
    with pytest.raises(ValueError):
        _convert_str_to_int(d, 5)
    with pytest.raises(ValueError):
        _convert_str_to_int(d, "hello")


PYGMO_ALGORITHMS = sorted(name for name in ALL_ALGORITHMS if name.startswith("pygmo_"))


@pytest.mark.skipif(not IS_PYGMO_INSTALLED, reason="pygmo is not installed.")
@pytest.mark.parametrize("algorithm", PYGMO_ALGORITHMS)
def test_seed_makes_runs_reproducible(algorithm):
    algo = ALL_ALGORITHMS[algorithm]().with_option(seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algo = algo.with_option_if_applicable(stopping_maxiter=10)

    def run():
        return minimize(
            fun=lambda x: x @ x,
            params=np.array([1.0, 2.0]),
            algorithm=algo,
            bounds=Bounds(lower=np.full(2, -5.0), upper=np.full(2, 5.0)),
        ).params

    np.testing.assert_array_equal(run(), run())

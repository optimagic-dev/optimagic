"""Test all available algorithms on a simple sum of squares function.

- only minimize
- only numerical derivative

"""

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from optimagic import mark
from optimagic.algorithms import AVAILABLE_ALGORITHMS, GLOBAL_ALGORITHMS
from optimagic.optimization.optimize import minimize
from optimagic.parameters.bounds import Bounds

AVAILABLE_LOCAL_ALGORITHMS = [
    name
    for name, algo in AVAILABLE_ALGORITHMS.items()
    if name not in GLOBAL_ALGORITHMS and name != "bhhh"
]

AVAILABLE_GLOBAL_ALGORITHMS = [
    name for name, algo in AVAILABLE_ALGORITHMS.items() if name in GLOBAL_ALGORITHMS
]

AVAILABLE_BOUNDED_ALGORITHMS = [
    name
    for name, algo in AVAILABLE_ALGORITHMS.items()
    if algo.algo_info.supports_infinite_bounds
]

BOUNDED_LOCAL_ALGORITHMS = [
    name
    for name, algo in AVAILABLE_ALGORITHMS.items()
    if name not in AVAILABLE_LOCAL_ALGORITHMS and algo.algo_info.supports_bounds
]


@pytest.fixture
def algo(algorithm):
    return AVAILABLE_ALGORITHMS[algorithm]


def _get_seed(algo):
    return {"seed": 12345} if hasattr(algo, "seed") else {}


def _get_required_decimals(algo):
    return (
        1 if algo.algo_info.is_global else 3
    )  # only scipy_trustconstr fails with 4 decimals


@mark.least_squares
def sos(x):
    return x


def _get_params_and_binding_bounds(algo):
    # these are binding bounds
    params = np.array([3, 2, -3])
    bounds = Bounds(
        lower=np.array([1, -np.inf, -np.inf]), upper=np.array([np.inf, np.inf, -1])
    )
    expected = np.array([1, 0, -1])
    return params, bounds, expected


# Tests all algorithms with binding bounds
@pytest.mark.parametrize("algorithm", AVAILABLE_BOUNDED_ALGORITHMS)
def test_sum_of_squares_with_binding_bounds(algorithm, algo):
    params, bounds, expected = _get_params_and_binding_bounds(algo)
    algo_options = _get_seed(algo)
    decimal = _get_required_decimals(algo)

    res = minimize(
        fun=sos,
        params=params,
        bounds=bounds,
        algorithm=algorithm,
        collect_history=True,
        algo_options=algo_options,
        skip_checks=True,
    )
    assert res.success in [True, None]
    aaae(res.params, expected, decimal=decimal)


def _get_params_and_bounds_on_local(algo):
    params = np.arange(3)
    bounds = None
    expected = np.zeros(3)
    if algo.algo_info.needs_bounds:
        # what bounds to have?
        bounds = Bounds(lower=np.full(3, -10), upper=np.full(3, 10))
    return params, bounds, expected


# Test all local algorithms without bounds unless needed
@pytest.mark.parametrize("algorithm", AVAILABLE_LOCAL_ALGORITHMS)
def test_sum_of_squares_on_local_algorithms(algorithm, algo):
    params, bounds, expected = _get_params_and_bounds_on_local(algo)
    algo_options = _get_seed(algo)
    decimal = _get_required_decimals(algo)

    res = minimize(
        fun=sos,
        params=params,
        bounds=bounds,
        algorithm=algorithm,
        collect_history=True,
        algo_options=algo_options,
        skip_checks=True,
    )
    assert res.success in [True, None]
    aaae(res.params, expected, decimal=decimal)


def _get_params_and_bounds_on_global(algo):
    params = np.array([0.35, 0.35])
    bounds = Bounds(lower=np.array([0.2, -0.5]), upper=np.array([1, 0.5]))
    expected = np.array([0.2, 0])
    return params, bounds, expected


# Test all global algorithms with bounds and local algorithms with bounds
@pytest.mark.parametrize(
    "algorithm", AVAILABLE_GLOBAL_ALGORITHMS + BOUNDED_LOCAL_ALGORITHMS
)
def test_sum_of_squares_on_global_algorithms(algorithm, algo):
    params, bounds, expected = _get_params_and_bounds_on_global(algo)
    algo_options = _get_seed(algo)
    decimal = _get_required_decimals(algo)

    res = minimize(
        fun=sos,
        params=params,
        bounds=bounds,
        algorithm=algorithm,
        collect_history=True,
        algo_options=algo_options,
        skip_checks=True,
    )
    assert res.success in [True, None]
    aaae(res.params, expected, decimal=decimal)

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal as aae

from optimagic.parameters.bounds import Bounds
from optimagic.parameters.tree_conversion import get_tree_converter
from optimagic.typing import AggregationLevel


@pytest.fixture()
def params():
    df = pd.DataFrame({"value": [3, 4], "lower_bound": [0, 0]}, index=["c", "d"])
    params = ([0, np.array([1, 2]), {"a": df, "b": 5}], 6)
    return params


@pytest.fixture()
def upper_bounds():
    upper = ([None, np.array([11, np.inf]), None], 100)
    return upper


FUNC_EVALS = [
    5.0,
    np.float32(5),
    np.ones(5),
    {"a": 1, "b": 2, "c": [np.full(4, 0.5)]},
    pd.Series(1, index=list("abcde")),
    np.ones(5),
    {"a": 1, "b": 2},
]


@pytest.mark.parametrize("func_eval", FUNC_EVALS)
def test_tree_converter_scalar_solver(params, upper_bounds, func_eval):
    bounds = Bounds(
        upper=upper_bounds,
    )
    converter, flat_params = get_tree_converter(
        params=params,
        bounds=bounds,
        func_eval=func_eval,
        derivative_eval=params,
        solver_type=AggregationLevel.SCALAR,
    )

    expected_values = np.arange(7)
    expected_lb = np.array([-np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf])
    expected_ub = np.array([np.inf, 11, np.inf, np.inf, np.inf, np.inf, 100])
    expected_names = ["0_0", "0_1_0", "0_1_1", "0_2_a_c", "0_2_a_d", "0_2_b", "1"]

    aae(flat_params.values, expected_values)
    aae(flat_params.lower_bounds, expected_lb)
    aae(flat_params.upper_bounds, expected_ub)
    assert flat_params.names == expected_names

    aae(converter.params_flatten(params), np.arange(7))
    unflat = converter.params_unflatten(np.arange(7))
    assert unflat[0][0] == params[0][0]
    aae(unflat[0][1], params[0][1])


SOLVER_TYPES = [
    AggregationLevel.SCALAR,
    AggregationLevel.LIKELIHOOD,
    AggregationLevel.LEAST_SQUARES,
]


@pytest.mark.parametrize("solver_type", SOLVER_TYPES)
def test_tree_conversion_fast_path(solver_type):
    if solver_type == AggregationLevel.SCALAR:
        derivative_eval = np.arange(3) * 2
        func_eval = 3
    else:
        derivative_eval = np.arange(6).reshape(2, 3)
        func_eval = np.ones(2)

    converter, flat_params = get_tree_converter(
        params=np.arange(3),
        bounds=None,
        func_eval=func_eval,
        derivative_eval=derivative_eval,
        solver_type=solver_type,
    )

    aae(flat_params.values, np.arange(3))
    aae(flat_params.lower_bounds, np.full(3, -np.inf))
    aae(flat_params.upper_bounds, np.full(3, np.inf))
    assert flat_params.names == list(map(str, range(3)))

    aae(converter.params_flatten(np.arange(3)), np.arange(3))
    aae(converter.params_unflatten(np.arange(3)), np.arange(3))
    aae(converter.derivative_flatten(derivative_eval), derivative_eval)

import numpy as np
import pandas as pd
import pytest
from estimagic.parameters.tree_conversion import get_tree_converter
from numpy.testing import assert_array_equal as aae
from pybaum import tree_equal


@pytest.fixture
def params():
    df = pd.DataFrame({"value": [3, 4], "lower_bound": [0, 0]}, index=["c", "d"])
    params = ([0, np.array([1, 2]), {"a": df, "b": 5}], 6)
    return params


@pytest.fixture
def upper_bounds():
    df = pd.DataFrame({"value": [10, 10], "lower_bound": [0, 0]}, index=["c", "d"])
    upper = ([np.inf, np.array([11, np.inf]), {"a": df, "b": np.inf}], 100)
    return upper


def test_tree_converter_no_constraints_scalar_func(params, upper_bounds):
    converter, flat_params = get_tree_converter(
        params=params,
        lower_bounds=None,
        upper_bounds=upper_bounds,
        func_eval=5,
        derivative_eval=params,
        primary_key="value",
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
    assert tree_equal(converter.params_unflatten(np.arange(7)), params)

    assert converter.func_flatten(3) == 3
    assert isinstance(converter.func_flatten(3), float)


# dict func with tree key
# dict func with dict key

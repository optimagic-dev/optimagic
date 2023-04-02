import pytest
import numpy as np
from estimagic.exceptions import InvalidParamsError
from estimagic.parameters.constraint_tools import check_constraints, count_free_params
from estimagic.parameters.check_constraints import _iloc


def test_count_free_params_no_constraints():
    params = {"a": 1, "b": 2, "c": [3, 3]}
    assert count_free_params(params) == 4


def test_count_free_params_with_constraints():
    params = {"a": 1, "b": 2, "c": [3, 3]}
    constraints = [{"selector": lambda x: x["c"], "type": "equality"}]
    assert count_free_params(params, constraints) == 3


def test_check_constraints():
    params = {"a": 1, "b": 2, "c": [3, 4]}
    constraints = [{"selector": lambda x: x["c"], "type": "equality"}]

    with pytest.raises(InvalidParamsError):
        check_constraints(params, constraints)


def test_iloc():
    # Case 1: ignore_first_row = True
    dictionary = {
        "index": ["a", "b", "c"],
        "lower_bounds": [0, 0, 0],
        "upper_bounds": [1, 1, 1],
        "is_fixed_to_value": [False, False, True],
    }
    info = [0, 2]
    ignore_first_row = True
    expected_result = {
        "lower_bounds": np.array([0]),
        "upper_bounds": np.array([1]),
        "is_fixed_to_value": np.array([True]),
    }
    result = _iloc(dictionary, info, ignore_first_row)
    assert len(result) == len(expected_result)
    for k, v in expected_result.items():
        assert k in result
        print(result[k], v)
        assert np.array_equal(result[k], v)

    # Case 2: ignore_first_row = False
    dictionary = {
        "index": ["a", "b", "c"],
        "lower_bounds": [0, 0, 0],
        "upper_bounds": [1, 1, 1],
        "is_fixed_to_value": [False, False, True],
    }
    info = [0, 2]
    ignore_first_row = False
    expected_result = {
        "lower_bounds": np.array([0, 0]),
        "upper_bounds": np.array([1, 1]),
        "is_fixed_to_value": np.array([False, True]),
    }
    result = _iloc(dictionary, info, ignore_first_row)
    assert len(result) == len(expected_result)
    for k, v in expected_result.items():
        assert k in result
        assert np.array_equal(result[k], v)

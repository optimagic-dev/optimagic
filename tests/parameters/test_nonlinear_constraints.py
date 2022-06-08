import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from estimagic.parameters.nonlinear_constraints import (
    _check_validity_nonlinear_constraint,
)
from estimagic.parameters.nonlinear_constraints import _get_positivity_transform
from estimagic.parameters.nonlinear_constraints import _get_transformation_type
from estimagic.parameters.nonlinear_constraints import _process_selector
from estimagic.parameters.nonlinear_constraints import (
    equality_as_inequality_constraints,
)
from estimagic.parameters.nonlinear_constraints import process_nonlinear_constraints
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal


########################################################################################
# _get_transformation_type
########################################################################################
TEST_CASES = [
    (0, np.inf, "identity"),  # (lower_bounds, upper_bounds, expected)
    (-1, 2, "stack"),
    (np.zeros(3), np.ones(3), "stack"),
    (np.zeros(3), np.tile(np.inf, 3), "identity"),
    (np.array([1, 2]), np.tile(np.inf, 2), "subtract_lb"),
]


@pytest.mark.parametrize("lower_bounds, upper_bounds, expected", TEST_CASES)
def test_get_transformation_type(lower_bounds, upper_bounds, expected):
    got = _get_transformation_type(lower_bounds, upper_bounds)
    assert got == expected


########################################################################################
# _process_selector
########################################################################################
TEST_CASES = [
    ({"selector": lambda x: x**2}, 10, 100),  # (constraint, params, expected)
    ({"loc": "a"}, pd.Series([0, 1], index=["a", "b"]), 0),
    (
        {"query": "a == 1"},
        pd.DataFrame([[1], [0]], columns=["a"]),
        pd.DataFrame([[1]], columns=["a"]),
    ),
]


@pytest.mark.parametrize("constraint, params, expected", TEST_CASES)
def test_process_selector(constraint, params, expected):
    _selector = _process_selector(constraint)
    got = _selector(params)

    if isinstance(got, pd.DataFrame):
        assert_frame_equal(got, expected)
    else:
        assert got == expected


########################################################################################
# _check_validity_nonlinear_constraint
########################################################################################
TEST_CASES = [
    {},  # no fun
    {"func": 10},  # non-callable fun
    {"func": lambda x: x, "derivative": 10},  # non-callable jac
    {"func": lambda x: x},  # no bounds at all
    {
        "func": lambda x: x,
        "value": 1,
        "lower_bounds": 1,
    },  # cannot have value and bounds
    {
        "func": lambda x: x,
        "value": 1,
        "upper_bounds": 1,
    },  # cannot have value and bounds
    {"func": lambda x: x},  # needs to have at least one bound
    {"func": lambda x: x, "lower_bounds": 1, "upper_bounds": 0},
    {"func": lambda x: x, "selector": 10},
    {"func": lambda x: x, "loc": 10},
    {"func": lambda x: x, "query": 10},
]

TEST_CASES = list(
    itertools.product(TEST_CASES, [np.arange(3), pd.DataFrame({"a": [0, 1, 2]})])
)


@pytest.mark.parametrize("constraint, params", TEST_CASES)
def test_check_validity_nonlinear_constraint(constraint, params):
    with pytest.raises(ValueError):
        _check_validity_nonlinear_constraint(constraint, params, skip_checks=False)


def test_check_validity_nonlinear_constraint_correct_example():
    constr = {
        "func": lambda x: x,
        "derivative": lambda x: np.ones_like(x),
        "lower_bounds": np.arange(4),
        "selector": lambda x: x[:1],
    }
    _check_validity_nonlinear_constraint(constr, params=np.arange(4))


########################################################################################
# equality_as_inequality_constraints
########################################################################################
TEST_CASES = [
    (
        [
            {
                "type": "ineq",
                "fun": lambda x: np.array([x]),
                "jac": lambda x: np.array([[1]]),
                "n_constr": 1,
            }
        ],  # constraints
        "same",  # expected
    ),
    (
        [
            {
                "type": "ineq",
                "fun": lambda x: np.array([x]),
                "jac": lambda x: np.array([[1]]),
                "n_constr": 1,
            }
        ],  # constraints
        [
            {
                "type": "eq",
                "fun": lambda x: np.array([x, -x]).reshape(-1, 1),
                "jac": lambda x: np.array([[1], [-1]]),
                "n_constr": 1,
            }
        ],  # expected
    ),
]


@pytest.mark.parametrize("constraints, expected", TEST_CASES)
def test_equality_as_inequality_constraints(constraints, expected):
    got = equality_as_inequality_constraints(constraints)
    if expected == "same":
        assert got == constraints

    for g, c in zip(got, constraints):
        if c["type"] == "eq":
            assert g["n_constr"] == 2 * c["n_constr"]
        assert g["type"] == "ineq"


########################################################################################
# _get_positivity_transform
########################################################################################
TEST_CASES = [
    #  (pis_pos_constr, lower_bounds, upper_bounds, case, expected)  # noqa: E800
    ("stack", 0, 0, "fun", np.array([1, -1])),
    ("stack", 1, 1, "fun", np.array([0, 0])),
    ("stack", 0, 0, "jac", np.array([1, -1])),
    ("stack", 1, 1, "jac", np.array([1, -1])),
    ("identity", 0, 1, "fun", np.array([1])),
    ("identity", 1, 0, "jac", np.array([1])),
]


@pytest.mark.parametrize(
    "is_pos_constr, lower_bounds, upper_bounds, case, expected", TEST_CASES
)
def test_get_positivity_transform(
    is_pos_constr, lower_bounds, upper_bounds, case, expected
):
    transform = _get_positivity_transform(
        is_pos_constr, lower_bounds, upper_bounds, case
    )
    got = transform(np.array([1]))
    assert np.all(got == expected)


########################################################################################
# process_nonlinear_constraints
########################################################################################


def test_process_nonlinear_constraints():

    nonlinear_constraints = [
        {"type": "nonlinear", "func": lambda x: np.dot(x, x), "value": 1},
        {
            "type": "nonlinear",
            "func": lambda x: x,
            "lower_bounds": -1,
            "upper_bounds": 2,
        },
    ]

    params = np.array([1.0])

    @dataclass
    class Converter:
        def params_from_internal(self, x):
            return x

    converter = Converter()

    got = process_nonlinear_constraints(
        nonlinear_constraints, params, converter, None, None, None, False
    )

    expected = [
        {"type": "eq", "func": lambda x: np.dot(x, x) - 1.0, "n_constr": 1},
        {
            "type": "ineq",
            "func": lambda x: np.concatenate((x + 1.0, 2.0 - x), axis=0),
            "n_constr": 2,
        },
    ]

    for g, e in zip(got, expected):
        assert g["type"] == e["type"]
        assert g["n_constr"] == e["n_constr"]
        for x in [0.1, 0.2, 1.2, -2.0]:
            x = np.array([x])
            assert_array_equal(g["func"](x), e["func"](x))
        assert "derivative" in g
        assert "tol" in g

"""Test the pc processing."""
import numpy as np
import pandas as pd
import pytest

from estimagic.optimization.process_constraints import _apply_constraint_killers
from estimagic.optimization.process_constraints import _process_selectors
from estimagic.optimization.process_constraints import (
    _replace_pairwise_equality_by_equality,
)
from estimagic.optimization.process_constraints import process_constraints
from estimagic.tests.optimization.test_reparametrize import reduce_params

constr1 = {"loc": 0, "type": "equality"}
expected1 = {"index": [0, 1, 2]}


constr2 = {"loc": 2, "type": "fixed"}
expected2 = {"index": [6, 8]}

constr3 = {"locs": [0, 1], "type": "pairwise_equality"}
expected3 = {"indices": [[0, 1, 2], [3, 4, 5]]}

test_cases = [(constr1, expected1), (constr2, expected2), (constr3, expected3)]


@pytest.mark.parametrize("constraint, expected", test_cases)
def test_process_selectors(constraint, expected):
    ind_tups = [
        (0, "a"),
        (0, "b"),
        (0, "c"),
        (1, "f"),
        (1, "e"),
        (1, "d"),
        (2, "a"),
        (3, "a"),
        (2, "b"),
    ]
    params_df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(ind_tups),
        columns=["bla", "blubb"],
        data=np.ones((9, 2)),
    )

    calculated = _process_selectors([constraint], params_df)[0]

    if constraint["type"] != "pairwise_equality":
        indices = [calculated["index"]]
        expected_indices = [expected["index"]]
    else:
        indices = calculated["indices"]
        expected_indices = expected["indices"]

    for calc, exp in zip(indices, expected_indices):
        for ind_tup1, ind_tup2 in zip(calc, exp):
            assert ind_tup1 == ind_tup2


def test_replace_pairwise_equality_by_equality():
    constr = {"indices": [[0, 1], [2, 3]], "type": "pairwise_equality"}

    expected = [
        {"index": [0, 2], "type": "equality"},
        {"index": [1, 3], "type": "equality"},
    ]

    calculated = _replace_pairwise_equality_by_equality([constr])

    assert calculated == expected


def test_apply_killer_constraints():
    constraints = [
        {"loc": "a", "type": "sdcorr"},
        {"loc": "b", "type": "sum", "id": "bla"},
        {"loc": "c", "type": "sum", "id": 3},
        {"kill": 3},
    ]

    calculated = _apply_constraint_killers(constraints)
    expected = [
        {"loc": "a", "type": "sdcorr"},
        {"loc": "b", "type": "sum", "id": "bla"},
    ]

    assert calculated == expected


def test_apply_killer_constraint_invalid():
    constraints = [{"loc": "a"}, {"loc": "b", "id": 5}, {"kill": 6}]
    with pytest.raises(KeyError):
        _apply_constraint_killers(constraints)


invalid_cases = [
    "basic_probability",
    "uncorrelated_covariance",
    "basic_covariance",
    "basic_increasing",
    "basic_equality",
    "query_equality",
    "basic_sdcorr",
]


@pytest.mark.parametrize("case", invalid_cases)
def test_valuep_error_if_constraints_are_violated(
    example_params, all_constraints, case
):
    constraints = all_constraints[case]
    params = reduce_params(example_params, constraints)
    for val in ["invalid_value0", "invalid_value1"]:
        params["value"] = params[val]

        with pytest.raises(ValueError):
            process_constraints(constraints, params)


def test_invalid_bound_for_increasing():
    params = pd.DataFrame(data=[[1], [2], [2.9]], columns=["value"])
    params["lower"] = [-np.inf, 1, 0.5]
    params["upper"] = np.nan

    constraints = [{"loc": params.index, "type": "increasing"}]

    with pytest.raises(ValueError):
        process_constraints(constraints, params)


def test_one_bound_is_allowed_for_increasing():
    params = pd.DataFrame(data=[[1], [2], [2.9]], columns=["value"])
    params["lower"] = [-np.inf, 1, -np.inf]
    params["upper"] = [np.inf, 2, np.inf]

    constraints = [{"loc": params.index, "type": "increasing"}]

    process_constraints(constraints, params)

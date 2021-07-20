"""Test the pc processing."""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from estimagic.parameters.process_constraints import _process_selectors
from estimagic.parameters.process_constraints import (
    _replace_pairwise_equality_by_equality,
)
from estimagic.parameters.process_constraints import process_constraints
from estimagic.tests.parameters.test_reparametrize import reduce_params

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
    params["lower_bound"] = [-np.inf, 1, 0.5]
    params["upper_bound"] = np.nan

    constraints = [{"loc": params.index, "type": "increasing"}]

    with pytest.raises(ValueError):
        process_constraints(constraints, params)


def test_one_bound_is_allowed_for_increasing():
    params = pd.DataFrame(data=[[1], [2], [2.9]], columns=["value"])
    params["lower_bound"] = [-np.inf, 1, -np.inf]
    params["upper_bound"] = [np.inf, 2, np.inf]

    constraints = [{"loc": params.index, "type": "increasing"}]

    process_constraints(constraints, params)


EMPTY_CONSTRAINTS = [
    [{"loc": [], "type": "covariance"}],
    [{"query": "value != value", "type": "sdcorr"}],
    [{"locs": [[], []], "type": "pairwise_equality"}],
    [{"queries": ["value != value"] * 3, "type": "pairwise_equality"}],
]


@pytest.mark.parametrize("constraints", EMPTY_CONSTRAINTS)
def test_empty_constraint_is_dropped(constraints):
    params = pd.DataFrame(np.ones((5, 1)), columns=["value"])
    pc, pp = process_constraints(constraints, params)
    # no transforming constraints
    assert pc == []
    # pre-replacements are just copying the parameter vector
    aaae(pp["_pre_replacements"], np.arange(5))
    # no post replacements
    aaae(pp["_post_replacements"], np.full(5, -1))

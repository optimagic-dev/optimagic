"""Tests for the conversion of deprecated dictionary constraints to Constraint objects.

pre_process_constraints converts dictionary constraints (including the deprecated
loc, locs, query and queries selection fields) into constraint objects. These tests
verify that the converted constraints select exactly the same parameters as the old
dictionary processing did, for every combination of selection field and params format
that used to be supported.

"""

import numpy as np
import pandas as pd
import pytest

import optimagic as om
from optimagic.deprecations import FixedValueConstraint, pre_process_constraints
from optimagic.exceptions import InvalidConstraintError
from optimagic.parameters.conversion import get_converter
from optimagic.typing import AggregationLevel


def _free_mask(params, constraints, bounds=None):
    _, internal = get_converter(
        params=params,
        constraints=pre_process_constraints(constraints),
        bounds=bounds,
        func_eval=1.0,
        solver_type=AggregationLevel.SCALAR,
    )
    return internal.free_mask.tolist()


@pytest.fixture()
def df_params():
    df = pd.DataFrame({"value": np.arange(6) + 10.0}, index=list("abcdef"))
    df.index.name = "name"
    return df


# ======================================================================================
# Conversion to the correct constraint objects
# ======================================================================================


def test_fixed_dict_is_converted():
    selector = lambda x: x[0]  # noqa: E731
    got = pre_process_constraints([{"type": "fixed", "selector": selector}])
    assert got == [om.FixedConstraint(selector=selector)]


def test_fixed_dict_with_value_is_converted():
    selector = lambda x: x[0]  # noqa: E731
    got = pre_process_constraints([{"type": "fixed", "selector": selector, "value": 5}])
    assert got == [FixedValueConstraint(selector=selector, value=5)]
    assert got[0]._to_dict() == {"type": "fixed", "selector": selector, "value": 5}


def test_covariance_dict_with_regularization_is_converted():
    selector = lambda x: x[:6]  # noqa: E731
    got = pre_process_constraints(
        [{"type": "covariance", "selector": selector, "regularization": 0.1}]
    )
    assert got == [om.FlatCovConstraint(selector=selector, regularization=0.1)]


def test_linear_dict_is_converted():
    selector = lambda x: x[[0, 1]]  # noqa: E731
    got = pre_process_constraints(
        [{"type": "linear", "selector": selector, "weights": 1, "upper_bound": 5}]
    )
    assert got == [om.LinearConstraint(selector=selector, weights=1, upper_bound=5)]


def test_nonlinear_dict_bounds_are_renamed():
    selector = lambda x: x[[0, 1]]  # noqa: E731
    func = lambda x: x @ x  # noqa: E731
    got = pre_process_constraints(
        [
            {
                "type": "nonlinear",
                "selector": selector,
                "func": func,
                "lower_bounds": 1,
                "upper_bounds": 2,
            }
        ]
    )
    assert got == [
        om.NonlinearConstraint(
            selector=selector, func=func, lower_bound=1, upper_bound=2
        )
    ]


def test_nonlinear_dict_without_selection_field_selects_all_params():
    func = lambda x: x @ x  # noqa: E731
    got = pre_process_constraints([{"type": "nonlinear", "func": func, "value": 1}])
    params = np.arange(3)
    assert got[0].selector(params) is params


def test_nonlinear_dict_loc_selector_has_no_value_indexing(df_params):
    # the nonlinear pipeline applied .loc without selecting the value column
    got = pre_process_constraints(
        [
            {
                "type": "nonlinear",
                "loc": ["b", "e"],
                "func": lambda x: x.sum(),
                "value": 1,
            }
        ]
    )
    selected = got[0].selector(df_params)
    pd.testing.assert_frame_equal(selected, df_params.loc[["b", "e"]])


def test_pairwise_equality_dict_with_selectors_is_converted():
    selectors = [lambda x: x[0], lambda x: x[1]]
    got = pre_process_constraints(
        [{"type": "pairwise_equality", "selectors": selectors}]
    )
    assert got == [om.PairwiseEqualityConstraint(selectors=selectors)]


# ======================================================================================
# loc / locs / query / queries select the same positions as before
# ======================================================================================


def test_loc_on_numpy_params():
    constraints = [{"type": "fixed", "loc": [1, 4]}]
    assert _free_mask(np.arange(6) + 10.0, constraints) == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]


def test_locs_on_numpy_params():
    # the pairing implies x1 == x0 and x4 == x3, which must hold at the start values
    params = np.array([5.0, 5.0, 2.0, 7.0, 7.0, 3.0])
    constraints = [{"type": "pairwise_equality", "locs": [[1, 4], [0, 3]]}]
    assert _free_mask(params, constraints) == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]


def test_loc_on_series_params():
    params = pd.Series(np.arange(6) + 10.0, index=list("abcdef"))
    constraints = [{"type": "fixed", "loc": ["b", "e"]}]
    assert _free_mask(params, constraints) == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]


def test_loc_on_dataframe_params(df_params):
    constraints = [{"type": "fixed", "loc": ["b", "e"]}]
    assert _free_mask(df_params, constraints) == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]


def test_scalar_loc_on_dataframe_params(df_params):
    constraints = [{"type": "fixed", "loc": "b"}]
    assert _free_mask(df_params, constraints) == [
        True,
        False,
        True,
        True,
        True,
        True,
    ]


def test_query_on_dataframe_params(df_params):
    constraints = [{"type": "fixed", "query": "name == 'b' | name == 'e'"}]
    assert _free_mask(df_params, constraints) == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]


@pytest.fixture()
def paired_df_params():
    # values are chosen such that b == a and e == d, as required by the pairings in
    # the tests below
    df = pd.DataFrame(
        {"value": [10.0, 10.0, 12.0, 13.0, 13.0, 15.0]}, index=list("abcdef")
    )
    df.index.name = "name"
    return df


def test_locs_on_dataframe_params(paired_df_params):
    constraints = [{"type": "pairwise_equality", "locs": [["b", "e"], ["a", "d"]]}]
    assert _free_mask(paired_df_params, constraints) == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]


def test_queries_on_dataframe_params(paired_df_params):
    queries = ["name == 'b' | name == 'e'", "name == 'a' | name == 'd'"]
    constraints = [{"type": "pairwise_equality", "queries": queries}]
    assert _free_mask(paired_df_params, constraints) == [
        True,
        False,
        True,
        True,
        False,
        True,
    ]


def test_equality_dict_with_loc():
    df = pd.DataFrame(
        {"value": [10.0, 11.0, 12.0, 13.0, 11.0, 15.0]}, index=list("abcdef")
    )
    constraints = [{"type": "equality", "loc": ["b", "e"]}]
    assert _free_mask(df, constraints) == [
        True,
        True,
        True,
        True,
        False,
        True,
    ]


# ======================================================================================
# Invalid selection fields and misspecified dicts raise InvalidConstraintError
# ======================================================================================


def test_dict_without_type_raises():
    with pytest.raises(InvalidConstraintError):
        pre_process_constraints([{"selector": lambda x: x[0]}])


def test_dict_with_unknown_type_raises():
    with pytest.raises(InvalidConstraintError):
        pre_process_constraints([{"type": "banana", "selector": lambda x: x[0]}])


def test_dict_without_selection_field_raises():
    with pytest.raises(InvalidConstraintError):
        pre_process_constraints([{"type": "fixed"}])


def test_dict_with_too_many_selection_fields_raises():
    with pytest.raises(InvalidConstraintError):
        pre_process_constraints(
            [{"type": "fixed", "loc": [0], "selector": lambda x: x[0]}]
        )


@pytest.mark.parametrize("field", ["selectors", "queries", "locs"])
def test_plural_selection_fields_raise_for_single_selector_constraints(field):
    with pytest.raises(InvalidConstraintError):
        pre_process_constraints([{"type": "equality", field: None}])


@pytest.mark.parametrize("field", ["selector", "query", "loc"])
def test_singular_selection_fields_raise_for_pairwise_equality(field):
    with pytest.raises(InvalidConstraintError):
        pre_process_constraints([{"type": "pairwise_equality", field: None}])


def test_query_on_numpy_params_raises():
    constraints = [{"type": "fixed", "query": "a == 1"}]
    with pytest.raises(InvalidConstraintError):
        _free_mask(np.arange(6) + 10.0, constraints)


def test_loc_on_pytree_params_raises():
    constraints = [{"type": "fixed", "loc": "a"}]
    with pytest.raises(InvalidConstraintError):
        _free_mask({"a": 1.0, "b": 2.0}, constraints)


def test_query_on_series_params_raises():
    params = pd.Series(np.arange(6) + 10.0, index=list("abcdef"))
    constraints = [{"type": "fixed", "query": "index == 'b'"}]
    with pytest.raises(InvalidConstraintError):
        _free_mask(params, constraints)


def test_duplicates_in_locs_raise():
    constraints = [{"type": "pairwise_equality", "locs": [[1, 4], [0, 0]]}]
    with pytest.raises(InvalidConstraintError):
        _free_mask(np.arange(6) + 10.0, constraints)


def test_different_lengths_in_locs_raise():
    constraints = [{"type": "pairwise_equality", "locs": [[1, 4], [0, 3, 5]]}]
    with pytest.raises(InvalidConstraintError):
        _free_mask(np.arange(6) + 10.0, constraints)

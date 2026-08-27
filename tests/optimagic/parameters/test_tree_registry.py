import itertools
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_frame_equal

from optimagic.parameters.tree_registry import (
    leaf_names,
    tree_equal,
    tree_flatten,
    tree_leaves,
    tree_map,
    tree_unflatten,
)
from optimagic.typing import DEFAULT_NAMESPACE, OPTREE_NAMESPACES, VALUE_NAMESPACE


@pytest.fixture()
def value_df():
    df = pd.DataFrame(
        np.arange(6).reshape(3, 2),
        columns=["a", "value"],
        index=["alpha", "beta", "gamma"],
    )
    return df


@pytest.fixture()
def other_df():
    df = pd.DataFrame(index=["alpha", "beta", "gamma"])
    df["b"] = np.arange(3).astype(np.int16)
    df["c"] = 3.14
    return df


@pytest.fixture
def example_tree():
    return (
        [0, np.array([1, 2]), {"a": pd.Series([3, 4], index=["c", "d"]), "b": 5}],
        6,
    )


def test_flatten_df_with_value_column(value_df):
    flat, _ = tree_flatten(value_df, namespace=VALUE_NAMESPACE)
    assert flat == [1, 3, 5]


def test_unflatten_df_with_value_column(value_df):
    _, treedef = tree_flatten(value_df, namespace=VALUE_NAMESPACE)
    unflat = tree_unflatten(treedef, [10, 11, 12], namespace=VALUE_NAMESPACE)
    assert unflat.equals(value_df.assign(value=[10, 11, 12]))


def test_leaf_names_df_with_value_column(value_df):
    names = leaf_names(value_df, namespace=VALUE_NAMESPACE)
    assert names == ["alpha", "beta", "gamma"]


@pytest.mark.parametrize("namespace", [None, *OPTREE_NAMESPACES])
def test_leaf_names_of_namedtuple_use_field_names(namespace):
    class ParamsTuple(NamedTuple):
        alpha: float
        beta: float

    kwargs = {} if namespace is None else {"namespace": namespace}
    params = {"x": ParamsTuple(alpha=1.0, beta=2.0)}
    assert leaf_names(params, **kwargs) == ["x_alpha", "x_beta"]


def test_leaf_names_with_is_leaf():
    params = {"a": 1, "b": np.array([0, 1])}
    names = leaf_names(
        params,
        is_leaf=lambda tree: isinstance(tree, np.ndarray),
        namespace=VALUE_NAMESPACE,
    )
    expected_names = ["a", "b"]
    assert names == expected_names


def test_flatten_partially_numeric_df(other_df):
    flat, _ = tree_flatten(other_df, namespace=VALUE_NAMESPACE)
    assert flat == [0, 3.14, 1, 3.14, 2, 3.14]


def test_unflatten_partially_numeric_df(other_df):
    _, treedef = tree_flatten(other_df, namespace=VALUE_NAMESPACE)
    unflat = tree_unflatten(treedef, [1, 2, 3, 4, 5, 6], namespace=VALUE_NAMESPACE)
    other_df = other_df.assign(b=[1, 3, 5], c=[2, 4, 6])
    assert_frame_equal(unflat, other_df, check_dtype=False)


def test_leaf_names_partially_numeric_df(other_df):
    names = leaf_names(other_df, namespace=VALUE_NAMESPACE)
    assert names == ["alpha_b", "alpha_c", "beta_b", "beta_c", "gamma_b", "gamma_c"]


@pytest.fixture()
def bounds_df():
    return pd.DataFrame(
        {
            "value": [1, 2, 3],
            "lower_bound": [0, 0, 0],
            "upper_bound": [10, 20, 30],
            "soft_lower_bound": [0.5, 0.5, 0.5],
            "soft_upper_bound": [9, 19, 29],
        },
        index=["alpha", "beta", "gamma"],
    )


def test_tree_methods_with_default_namespace(bounds_df):
    leaves, treedef = tree_flatten(bounds_df)
    assert len(leaves) == 1
    assert_frame_equal(leaves[0], bounds_df)

    leaves = tree_leaves(bounds_df)
    assert len(leaves) == 1
    assert_frame_equal(leaves[0], bounds_df)

    tree = tree_unflatten(treedef, leaves)
    assert_frame_equal(tree, bounds_df)

    names = leaf_names(bounds_df)
    expected_names = [""]
    assert names == expected_names

    tree = tree_map(lambda x: x * 2, bounds_df)
    assert_frame_equal(tree, bounds_df * 2)


@pytest.mark.parametrize("namespace", OPTREE_NAMESPACES)
def test_tree_methods_with_registered_namespaces(namespace, bounds_df):
    expected_leaves = bounds_df[namespace].tolist()

    leaves, treedef = tree_flatten(bounds_df, namespace=namespace)
    assert leaves == expected_leaves

    leaves = tree_leaves(bounds_df, namespace=namespace)
    assert leaves == expected_leaves

    tree = tree_unflatten(treedef, leaves, namespace=namespace)
    assert_frame_equal(tree, bounds_df)

    names = leaf_names(bounds_df, namespace=namespace)
    assert names == ["alpha", "beta", "gamma"]

    tree = tree_map(lambda x: x * 2, bounds_df, namespace=namespace)
    doubled = [v * 2 for v in expected_leaves]
    expected = bounds_df.copy()
    expected[namespace] = doubled
    assert_frame_equal(tree, expected)


def test_tree_methods_raise_warning_with_unregisted_namespace():
    unregistered_namespace = "unregistered_namespace"
    tree, leaves = [0], [0]
    match_str = "is not registered."
    with pytest.warns(match=match_str):
        tree_flatten(tree, namespace=unregistered_namespace)
    with pytest.warns(match=match_str):
        tree_leaves(tree, namespace=unregistered_namespace)
    with pytest.warns(match=match_str):
        tree_unflatten(tree, leaves, namespace=unregistered_namespace)
    with pytest.warns(match=match_str):
        leaf_names(tree, namespace=unregistered_namespace)
    with pytest.warns(match=match_str):
        tree_map(lambda x: x * 2, tree, namespace=unregistered_namespace)


def test_tree_flatten_and_unflatten_with_None():
    params = [None]
    leaves, treespec = tree_flatten(params)
    assert leaves == []
    tree = tree_unflatten(treespec, leaves)
    assert tree == [None]


def test_leaf_names_with_none():
    names = leaf_names(None)
    assert names == []


@pytest.mark.parametrize("namespace", OPTREE_NAMESPACES + (DEFAULT_NAMESPACE,))
def test_dict_insertion_ordering_is_respected(namespace):
    params = {"b": [1, 4], "a": [8, 9]}
    leaves, _ = tree_flatten(params, namespace=namespace)
    assert leaves == [1, 4, 8, 9]

    tree = tree_unflatten(params, [1, 4, 8, 9], namespace=namespace)
    assert list(tree.items()) == [("b", [1, 4]), ("a", [8, 9])]

    leaves2 = tree_leaves(params, namespace=namespace)
    assert leaves2 == [1, 4, 8, 9]

    tree = tree_map(lambda x: x, params, namespace=namespace)
    assert list(tree.items()) == [("b", [1, 4]), ("a", [8, 9])]

    names = leaf_names(params, namespace=namespace)
    assert names == ["b_0", "b_1", "a_0", "a_1"]

    params = {"b": 8, "a": 5}
    counter = itertools.count()
    positions = tree_map(lambda _: next(counter), params, namespace=namespace)
    assert positions == {"b": 0, "a": 1}


def test_tree_equal_with_pandas_nodes_in_registered_namespace():
    tree = {
        "s": pd.Series([1.0, 2.0], index=["c", "d"]),
        "df": pd.DataFrame({"value": [1.0, 2.0]}, index=["i", "j"]),
    }
    copied = {
        "s": tree["s"].copy(),
        "df": tree["df"].copy(deep=True),
    }
    assert tree_equal(tree, copied, namespace=VALUE_NAMESPACE) is True


def test_tree_equal_detects_different_series_index():
    first = {"s": pd.Series([1.0], index=["x"])}
    second = {"s": pd.Series([1.0], index=["y"])}
    assert tree_equal(first, second, namespace=VALUE_NAMESPACE) is False


def test_tree_equal_with_unequal_values_and_structures():
    assert tree_equal({"a": 1.0}, {"a": 2.0}, namespace=VALUE_NAMESPACE) is False
    assert tree_equal({"a": 1.0}, {"b": 1.0}, namespace=VALUE_NAMESPACE) is False


def test_tree_equal_runs_raising_checkers_on_all_leaves():
    checkers = {np.ndarray: lambda x, y: aaae(x, y, decimal=5)}
    first = {"a": np.array([1.0]), "b": np.array([2.0])}
    second = {"a": np.array([1.0]), "b": np.array([99.0])}
    with pytest.raises(AssertionError):
        tree_equal(first, second, equality_checkers=checkers)


def test_tree_equal_returns_bool_with_none_returning_checkers():
    checkers = {np.ndarray: lambda x, y: aaae(x, y, decimal=5)}
    first = {"a": np.array([1.0]), "b": np.array([2.0])}
    second = {"a": np.array([1.0]), "b": np.array([2.0])}
    assert tree_equal(first, second, equality_checkers=checkers) is True

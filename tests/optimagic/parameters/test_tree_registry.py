import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from optimagic.parameters.tree_registry import (
    leaf_names,
    tree_flatten,
    tree_just_flatten,
    tree_map,
    tree_unflatten,
)
from optimagic.typing import OPTREE_NAMESPACES, VALUE_NAMESPACE


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


def test_tree_methods_with_empty_namespace(value_df):
    leaves, _ = tree_flatten(value_df)
    assert leaves == [value_df]

    leaves = tree_just_flatten(value_df)
    assert leaves == [value_df]

    names = leaf_names(value_df)
    expected_names = [""]
    assert names == expected_names

    tree = tree_map(lambda x: x * 2, value_df)
    assert_frame_equal(tree, value_df)


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


@pytest.mark.parametrize("namespace", OPTREE_NAMESPACES)
def test_tree_methods_with_optimagic_namespace(namespace, bounds_df):
    expected_leaves = bounds_df[namespace].tolist()

    leaves, _ = tree_flatten(bounds_df, namespace=namespace)
    assert leaves == expected_leaves

    leaves = tree_just_flatten(bounds_df, namespace=namespace)
    assert leaves == expected_leaves

    names = leaf_names(bounds_df, namespace=namespace)
    assert names == ["alpha", "beta", "gamma"]

    tree = tree_map(lambda x: x * 2, bounds_df, namespace=namespace)
    doubled = [v * 2 for v in expected_leaves]
    expected = bounds_df.copy()
    expected[namespace] = doubled
    assert_frame_equal(tree, expected)


@pytest.mark.parametrize(
    "tree_method",
    [tree_flatten, tree_just_flatten, leaf_names, tree_map, tree_flatten],
)
def test_tree_methods_raise_warning_with_unregisted_namespace(tree_method, value_df):
    """If namespace is not registered optree method fallbacks to default behaviour."""
    unregistered_namespace = "unregistered_namespace"
    with pytest.warns(match="is not registered."):
        if tree_method == tree_map:
            _ = tree_map(lambda x: x, value_df, namespace=unregistered_namespace)
        elif tree_method == tree_unflatten:
            _ = tree_method(value_df, [], namespace=unregistered_namespace)
        else:
            _ = tree_method(value_df, namespace=unregistered_namespace)


def test_tree_flatten_and_unflatten_with_None():
    params = [None]
    leaves, treespec = tree_flatten(params)
    assert leaves == []
    tree = tree_unflatten(treespec, leaves)
    assert tree == [None]


def test_dict_insertion_ordering_is_respected_for_registered_namespaces():
    params = {"b": [1, 4], "a": [8, 9]}
    leaves, _ = tree_flatten(params, namespace=VALUE_NAMESPACE)
    assert leaves == [1, 4, 8, 9]
    leaves2 = tree_just_flatten(params, namespace=VALUE_NAMESPACE)
    assert leaves2 == [1, 4, 8, 9]
    names = leaf_names(params, namespace=VALUE_NAMESPACE)
    assert names == ["b_0", "b_1", "a_0", "a_1"]


def test_dict_ordering_default_behaviour_is_by_name():
    params = {"b": [1, 4], "a": [8, 9]}
    leaves, _ = tree_flatten(params)
    assert leaves == [8, 9, 1, 4]

    leaves2 = tree_just_flatten(params)
    assert leaves2 == [8, 9, 1, 4]

    names = leaf_names(params)
    assert names == ["a_0", "a_1", "b_0", "b_1"]


def test_unflatten_respects_insertion_order():
    params = {"b": [1, 4], "a": [8, 9]}
    leaves, treespec = tree_flatten(params)
    tree = tree_unflatten(treespec, leaves)
    assert list(tree.items()) == [("b", [1, 4]), ("a", [8, 9])]
    leaves2, treespec2 = tree_flatten(params, namespace=VALUE_NAMESPACE)
    tree2 = tree_unflatten(treespec2, leaves2)
    assert list(tree2.items()) == [("b", [1, 4]), ("a", [8, 9])]


def test_map_always_respects_insertion_order():
    params = {"b": [1, 4], "a": [8, 9]}
    tree = tree_map(lambda x: x, params)
    assert list(tree.items()) == [("b", [1, 4]), ("a", [8, 9])]
    tree2 = tree_map(lambda x: x, params, namespace=VALUE_NAMESPACE)
    assert list(tree2.items()) == [("b", [1, 4]), ("a", [8, 9])]

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from optimagic.parameters.tree_registry import (
    extended,
    leaf_names,
    set_data_col_df_attribute,
    tree_flatten,
    tree_unflatten,
)


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


def test_flatten_df_with_value_column(value_df):
    flat, _ = tree_flatten(value_df, namespace=extended)
    assert flat == [1, 3, 5]


def test_unflatten_df_with_value_column(value_df):
    _, treedef = tree_flatten(value_df, namespace=extended)
    unflat = tree_unflatten(treedef, [10, 11, 12], namespace=extended)
    assert unflat.equals(value_df.assign(value=[10, 11, 12]))


def test_leaf_names_df_with_value_column(value_df):
    names = leaf_names(value_df, namespace=extended)
    assert names == ["alpha", "beta", "gamma"]


def test_flatten_partially_numeric_df(other_df):
    flat, _ = tree_flatten(other_df, namespace=extended)
    assert flat == [0, 3.14, 1, 3.14, 2, 3.14]


def test_unflatten_partially_numeric_df(other_df):
    _, treedef = tree_flatten(other_df, namespace=extended)
    unflat = tree_unflatten(treedef, [1, 2, 3, 4, 5, 6], namespace=extended)
    other_df = other_df.assign(b=[1, 3, 5], c=[2, 4, 6])
    assert_frame_equal(unflat, other_df, check_dtype=False)


def test_leaf_names_partially_numeric_df(other_df):
    names = leaf_names(other_df, namespace=extended)
    assert names == ["alpha_b", "alpha_c", "beta_b", "beta_c", "gamma_b", "gamma_c"]


def test_set_data_col_attribute_assigns_attribute(value_df):
    df = set_data_col_df_attribute(value_df, data_col="attr")
    assert df.attrs.get("data_col") == "attr"
    assert value_df.attrs.get("data_col") is None


def test_set_data_col_attribute_unflattened_tree_has_attribute(value_df):
    df = set_data_col_df_attribute(value_df, data_col="attr")
    tree, treedef = tree_flatten(df, namespace=extended)
    df = tree_unflatten(treedef, tree)
    assert df.attrs.get("data_col") == "attr"


def test_set_data_col_attribute_returns_nan(value_df):
    df = set_data_col_df_attribute(value_df, data_col="attr")
    tree, treedef = tree_flatten(df, namespace=extended)
    assert all(np.isnan(value) for value in tree)


def test_set_data_col_attribute_returs_column_values(value_df):
    df = set_data_col_df_attribute(value_df, data_col="a")
    tree, treedef = tree_flatten(df, namespace=extended)
    assert tree == [0, 2, 4]

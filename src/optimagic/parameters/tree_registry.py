"""Wrapper around pybaum get_registry to tailor it to optimagic."""

import itertools
from itertools import product

import numpy as np
import optree
import pandas as pd
from optree.pytree import PyTreeSpec

from optimagic.typing import extended_namespace


def _get_df_names(df):
    index_strings = list(df.index.map(_index_element_to_string))
    if "value" in df:
        out = index_strings
    else:
        out = ["_".join([loc, col]) for loc, col in product(index_strings, df.columns)]

    return out


def _index_element_to_string(element):
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry) for entry in element]
        res_string = "_".join(as_strings)
    else:
        res_string = str(element)

    return res_string


def tree_flatten(tree, is_leaf=None, namespace=""):
    with optree.dict_insertion_ordered(True, namespace=extended_namespace):
        return optree.tree_flatten(tree, is_leaf=is_leaf, namespace=namespace)


def tree_just_flatten(tree, is_leaf=None, namespace=""):
    leaves, _ = tree_flatten(tree, is_leaf=is_leaf, namespace=namespace)
    return leaves


extended = extended_namespace


def tree_unflatten(treedef, leaves, is_leaf=None, namespace=""):
    if not isinstance(treedef, PyTreeSpec):
        _, treedef = tree_flatten(treedef, is_leaf=is_leaf, namespace=namespace)
    return optree.tree_unflatten(treespec=treedef, leaves=leaves)


def tree_map(func, tree, is_leaf=None, namespace=""):
    return optree.tree_map(func, tree, is_leaf=is_leaf, namespace=namespace)


def leaf_names(tree, is_leaf=None, namespace="", separator="_"):
    _, treespec = tree_flatten(tree, is_leaf=is_leaf, namespace=namespace)
    paths = treespec.paths()
    return [separator.join(str(p) for p in path) for path in paths]


def set_data_col_df_attribute(tree, data_col):
    def set_attr(node):
        if isinstance(node, pd.DataFrame):
            node = node.copy()
            node.attrs["data_col"] = data_col
        return node

    return tree_map(set_attr, tree)


def _array_element_names(arr):
    dim_names = [map(str, range(n)) for n in arr.shape]
    names = list(map("_".join, itertools.product(*dim_names)))
    return names


def _flatten_df_optree(df):
    data_col = df.attrs.get("data_col", "value")
    is_value_df = "value" in df
    if is_value_df:
        flat = df.get(data_col, default=np.full(len(df), np.nan)).tolist()
    else:
        flat = df.to_numpy().flatten().tolist()

    aux_data = {
        "is_value_df": is_value_df,
        "df": df,
    }
    return flat, aux_data, _get_df_names(df)


def _unflatten_df_optree(aux_data, leaves):
    data_col = aux_data["df"].attrs.get("data_col", "value")
    if aux_data["is_value_df"]:
        out = aux_data["df"].assign(**{data_col: leaves})
    else:
        out = pd.DataFrame(
            data=np.array(leaves).reshape(aux_data["df"].shape),
            columns=aux_data["df"].columns,
            index=aux_data["df"].index,
        )
    return out


optree.register_pytree_node(
    pd.DataFrame,
    _flatten_df_optree,
    _unflatten_df_optree,
    namespace=extended_namespace,
)

optree.register_pytree_node(
    pd.Series,
    lambda sr: (
        sr.tolist(),
        {"index": sr.index, "name": sr.name},
        list(sr.index.map(_index_element_to_string)),
    ),
    lambda aux_data, leaves: pd.Series(leaves, **aux_data),
    namespace=extended_namespace,
)

optree.register_pytree_node(
    np.ndarray,
    lambda arr: (arr.flatten().tolist(), arr.shape, _array_element_names(arr)),
    lambda aux_data, leaves: np.array(leaves).reshape(aux_data),
    namespace=extended_namespace,
)

EQUALITY_CHECKERS = {}
EQUALITY_CHECKERS[np.ndarray] = lambda a, b: bool((a == b).all())
EQUALITY_CHECKERS[pd.Series] = lambda a, b: a.equals(b)
EQUALITY_CHECKERS[pd.DataFrame] = lambda a, b: a.equals(b)


def tree_equal(tree, other, is_leaf=None, namespace="", equality_checkers=None):
    equality_checkers = (
        EQUALITY_CHECKERS
        if equality_checkers is None
        else {**EQUALITY_CHECKERS, **equality_checkers}
    )

    first_flat, first_treespec = tree_flatten(
        tree, is_leaf=is_leaf, namespace=namespace
    )
    second_flat, second_treespec = tree_flatten(
        other, is_leaf=is_leaf, namespace=namespace
    )

    first_names = leaf_names(tree, is_leaf=is_leaf, namespace=namespace)
    second_names = leaf_names(tree, is_leaf=is_leaf, namespace=namespace)

    equal = first_names == second_names and first_treespec == second_treespec

    if equal:
        for first, second in zip(first_flat, second_flat, strict=True):
            check_func = equality_checkers.get(type(first), lambda a, b: a == b)
            equal = equal and check_func(first, second)
            if not equal:
                break

    return equal

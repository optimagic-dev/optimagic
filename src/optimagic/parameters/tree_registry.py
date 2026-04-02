"""Wrapper around optree to tailor it to optimagic."""

import itertools
from functools import partial
from itertools import product

import numpy as np
import optree
import pandas as pd
from optree.pytree import PyTreeSpec

from optimagic.typing import optree_namespaces

try:
    import jax.numpy as jnp  # type: ignore[import-not-found]

    _has_jax = True
except ImportError:
    _has_jax = False


EQUALITY_CHECKERS = {}
EQUALITY_CHECKERS[np.ndarray.__name__] = lambda a, b: bool((a == b).all())
EQUALITY_CHECKERS[pd.Series.__name__] = lambda a, b: a.equals(b)
EQUALITY_CHECKERS[pd.DataFrame.__name__] = lambda a, b: a.equals(b)

if _has_jax:
    EQUALITY_CHECKERS[jnp.ndarray.__name__] = lambda a, b: bool((a == b).all())


def tree_flatten(tree, is_leaf=None, namespace=""):
    if namespace:
        with optree.dict_insertion_ordered(True, namespace=namespace):
            return optree.tree_flatten(tree, is_leaf=is_leaf, namespace=namespace)
    else:
        return optree.tree_flatten(tree, is_leaf=is_leaf, namespace=namespace)


def tree_just_flatten(tree, is_leaf=None, namespace=""):
    leaves, _ = tree_flatten(tree, is_leaf=is_leaf, namespace=namespace)
    return leaves


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
    second_names = leaf_names(other, is_leaf=is_leaf, namespace=namespace)

    equal = first_names == second_names and first_treespec == second_treespec

    if equal:
        for first, second in zip(first_flat, second_flat, strict=True):
            check_func = equality_checkers.get(
                type(first).__name__, lambda a, b: a == b
            )
            equal = equal and check_func(first, second)
            if not equal:
                break

    return equal


def _flatten_df(df, data_col):
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


def _unflatten_df(aux_data, leaves, data_col):
    if aux_data["is_value_df"]:
        out = aux_data["df"].assign(**{data_col: leaves})
    else:
        out = pd.DataFrame(
            data=np.array(leaves).reshape(aux_data["df"].shape),
            columns=aux_data["df"].columns,
            index=aux_data["df"].index,
        )
    return out


def _flatten_series(series):
    return (
        series.tolist(),
        {"index": series.index, "name": series.name},
        list(series.index.map(_index_element_to_string)),
    )


def _unflatten_series(aux_data, leaves):
    return pd.Series(leaves, **aux_data)


def _flatten_ndarray(arr):
    return arr.flatten().tolist(), arr.shape, _array_element_names(arr)


def _unflatten_ndarray(aux_data, leaves):
    return np.array(leaves).reshape(aux_data)


if _has_jax:

    def _flatten_jax_array(arr):
        return arr.flatten().tolist(), arr.shape, _array_element_names(arr)

    def _unflatten_jax_array(aux_data, leaves):
        return jnp.array(leaves).reshape(aux_data)


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


def _array_element_names(arr):
    dim_names = [map(str, range(n)) for n in arr.shape]
    names = list(map("_".join, itertools.product(*dim_names)))
    return names


for namespace in optree_namespaces:
    optree.register_pytree_node(
        pd.DataFrame,
        partial(_flatten_df, data_col=namespace),
        partial(_unflatten_df, data_col=namespace),
        namespace=namespace,
    )

    optree.register_pytree_node(
        pd.Series,
        _flatten_series,
        _unflatten_series,
        namespace=namespace,
    )

    optree.register_pytree_node(
        np.ndarray,
        _flatten_ndarray,
        _unflatten_ndarray,
        namespace=namespace,
    )

    if _has_jax:
        optree.register_pytree_node(
            jnp.ndarray,
            _flatten_jax_array,
            _unflatten_jax_array,
            namespace=namespace,
        )

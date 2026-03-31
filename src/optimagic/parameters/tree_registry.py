"""Wrapper around pybaum get_registry to tailor it to optimagic."""

from collections import OrderedDict
from functools import partial
from itertools import product

import numpy as np
import optree
import pandas as pd
from optree.pytree import PyTreeSpec
from pybaum import get_registry as get_pybaum_registry


def get_registry(extended=False, data_col="value"):
    """Return pytree registry.

    Special Rules
    -------------
    If extended is True the registry contains pd.DataFrame. In optimagic a data frame
    can represent a 1d object with extra information, instead of a 2d object. This is
    only allowed for params data frames, in which case they contain a 'value' column.
    The extra information of such an object can be accessed using the data_col argument.
    By default the 'value' column is extracted. If data_col is not 'value' but the data
    frame contains a 'value' column, a list of np.nan is returned.

    Args:
        extended (bool): If True appends types 'numpy.ndarray', 'pandas.Series' and
            'pandas.DataFrame' to the registry.
        data_col (str): This column is used as the data source in a data frame when
            flattening and unflattening a pytree. Defaults to 'value'; see special rules
            above for behavior with non-default values.

    Returns:
        dict: The pytree registry.

    """
    types = (
        ["numpy.ndarray", "pandas.Series", "jax.numpy.ndarray"] if extended else None
    )
    registry = get_pybaum_registry(types=types)
    if extended:
        registry[pd.DataFrame] = {
            "flatten": partial(_flatten_df, data_col=data_col),
            "unflatten": partial(_unflatten_df, data_col=data_col),
            "names": _get_df_names,
        }
    return registry


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
    return flat, aux_data


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


extended = "extended"


def tree_flatten(tree, is_leaf=None, registry=None):
    if isinstance(tree, dict):
        tree = OrderedDict(tree)
    return optree.tree_flatten(
        tree, is_leaf=is_leaf, namespace=extended if registry else ""
    )


def tree_just_flatten(tree, is_leaf=None, registry=None):
    if isinstance(tree, dict):
        tree = OrderedDict(tree)

    return optree.tree_leaves(
        tree,
        is_leaf,
        namespace=extended if registry else "",
    )


def tree_unflatten(treedef, leaves, is_leaf=None, registry=None):
    if not isinstance(treedef, PyTreeSpec):
        if isinstance(treedef, dict):
            treedef = OrderedDict(treedef)
        _, treedef = optree.tree_flatten(
            treedef, namespace=extended if registry else ""
        )
    return optree.tree_unflatten(treespec=treedef, leaves=leaves)


def tree_map(func, tree, is_leaf=None, registry=None):
    return optree.tree_map(
        func, tree, is_leaf=is_leaf, namespace=extended if registry else ""
    )


def set_data_col_df_attribute(tree, data_col):
    def set_attr(node):
        if isinstance(node, pd.DataFrame):
            node = node.copy()
            node.attrs["data_col"] = data_col
        return node

    return tree_map(set_attr, tree)


def _flatten_df_optree(df):
    data_col = df.attrs.get("data_col", "value")
    return _flatten_df(df, data_col=data_col)


def _unflatten_df_optree(aux_data, leaves):
    data_col = aux_data["df"].attrs.get("data_col", "value")
    return _unflatten_df(aux_data=aux_data, leaves=leaves, data_col=data_col)


optree.register_pytree_node(
    pd.DataFrame,
    _flatten_df_optree,
    _unflatten_df_optree,
    namespace=extended,
)

optree.register_pytree_node(
    pd.Series,
    lambda sr: (
        sr.tolist(),
        {"index": sr.index, "name": sr.name},
    ),
    lambda aux_data, leaves: pd.Series(leaves, **aux_data),
    namespace=extended,
)

optree.register_pytree_node(
    np.ndarray,
    lambda arr: (arr.flatten().tolist(), arr.shape),
    lambda aux_data, leaves: np.array(leaves).reshape(aux_data),
    namespace=extended,
)
# DONT FORGET JAX

"""Wrapper around optree to tailor it to optimagic."""

import warnings
from functools import partial
from itertools import product

import numpy as np
import optree
import pandas as pd
from optree.pytree import PyTreeSpec

from optimagic.typing import DEFAULT_NAMESPACE, OPTREE_NAMESPACES

try:
    import jax.numpy as jnp  # type: ignore[import-not-found]
    import jaxlib  # type: ignore[import-not-found]

    _has_jax = True
except ImportError:
    _has_jax = False

_are_namespaces_registered = False


def tree_flatten(tree, is_leaf=None, namespace=DEFAULT_NAMESPACE):
    """Flatten a pytree."""
    _register_namespaces()
    _check_namespace(namespace)
    with optree.dict_insertion_ordered(True, namespace=namespace):
        return optree.tree_flatten(tree, is_leaf=is_leaf, namespace=namespace)


def tree_leaves(tree, is_leaf=None, namespace=DEFAULT_NAMESPACE):
    """Get the leaves of a pytree."""
    _register_namespaces()
    _check_namespace(namespace)
    with optree.dict_insertion_ordered(True, namespace=namespace):
        return optree.tree_leaves(tree, is_leaf=is_leaf, namespace=namespace)


def tree_unflatten(treedef, leaves, namespace=DEFAULT_NAMESPACE):
    """Reconstruct a pytree from the tree definition and the leaves."""
    _register_namespaces()

    if not isinstance(treedef, PyTreeSpec):
        _check_namespace(namespace)
        with optree.dict_insertion_ordered(True, namespace=namespace):
            treedef = optree.tree_structure(treedef, namespace=namespace)

    # Doesn't need to be wrapped with dict_insertion_ordered
    # because it keeps the insertion order for dictionaries by default.
    return optree.tree_unflatten(treedef, leaves)


def tree_map(func, tree, is_leaf=None, namespace=DEFAULT_NAMESPACE):
    """Map an input function over pytree args to produce a new pytree."""
    _register_namespaces()
    _check_namespace(namespace)

    # Doesn't need to be wrapped with dict_insertion_ordered
    # because it keeps the insertion order for dictionaries by default.
    return optree.tree_map(func, tree, is_leaf=is_leaf, namespace=namespace)


def leaf_names(tree, is_leaf=None, namespace=DEFAULT_NAMESPACE, separator="_"):
    """Get the path names for tree leaves."""
    _register_namespaces()
    _check_namespace(namespace)

    with optree.dict_insertion_ordered(True, namespace=namespace):
        paths, _, _ = optree.tree_flatten_with_path(
            tree, is_leaf=is_leaf, namespace=namespace
        )
    return [separator.join(str(p) for p in path) for path in paths]


def tree_equal(
    tree, other, is_leaf=None, namespace=DEFAULT_NAMESPACE, equality_checkers=None
):
    """Check the equality between two trees."""
    equality_checkers = (
        _get_equality_checkers()
        if equality_checkers is None
        else {**_get_equality_checkers(), **equality_checkers}
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


def _get_equality_checkers():
    """Return type-specific equality checkers for array and DataFrame leaves.

    These are used during pytree operations to compare leaves that don't
    support simple ``==`` equality (e.g. NumPy arrays, pandas objects).
    """
    equality_checkers = {}
    equality_checkers[np.ndarray.__name__] = lambda a, b: bool((a == b).all())
    equality_checkers[pd.Series.__name__] = lambda a, b: a.equals(b)
    equality_checkers[pd.DataFrame.__name__] = lambda a, b: a.equals(b)

    if _has_jax:
        equality_checkers[jnp.ndarray.__name__] = lambda a, b: bool((a == b).all())

    return equality_checkers


def _check_namespace(namespace: str) -> None:
    """Checks if the namespace is registered and raise a warning."""
    if namespace != DEFAULT_NAMESPACE and namespace not in OPTREE_NAMESPACES:
        warnings.warn(
            f"Namespace '{namespace}' is not registered. "
            f"Registered namespaces are: {','.join(OPTREE_NAMESPACES)}. "
            "Pytree method is being parsed with the default optree namespace."
        )


def _register_namespaces() -> None:
    """Register pytree flatten/unflatten methods for each namespace.

    This method must only be called once as each namespace must only be registered
    one time.
    """
    global _are_namespaces_registered  # noqa: PLW0603
    if _are_namespaces_registered is False:
        _are_namespaces_registered = True
        for namespace in OPTREE_NAMESPACES:
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
                    jaxlib._jax.ArrayImpl,
                    _flatten_jax_array,
                    _unflatten_jax_array,
                    namespace=namespace,
                )


def _flatten_df(df, data_col):
    """Flatten a dataframe."""
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
    """Reconstruct a dataframe."""
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
    """Flatten a series."""
    return (
        series.tolist(),
        {"index": series.index, "name": series.name},
        list(series.index.map(_index_element_to_string)),
    )


def _unflatten_series(aux_data, leaves):
    """Reconstruct a series."""
    return pd.Series(leaves, **aux_data)


def _flatten_ndarray(arr):
    """Flatten a numpy array."""
    return arr.flatten().tolist(), arr.shape, _array_element_names(arr)


def _unflatten_ndarray(aux_data, leaves):
    """Reconstrut a numpy array."""
    return np.array(leaves).reshape(aux_data)


if _has_jax:

    def _flatten_jax_array(arr):
        """Flatten a jax array."""
        return arr.flatten().tolist(), arr.shape, _array_element_names(arr)

    def _unflatten_jax_array(aux_data, leaves):
        """Unflatten a jax array."""
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
    names = list(map("_".join, product(*dim_names)))
    return names

"""Wrapper around optree to tailor it to optimagic."""

import contextlib
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, get_args

import numpy as np
import optree
import pandas as pd
from optree.pytree import PyTreeSpec

from optimagic.config import IS_JAX_INSTALLED
from optimagic.typing import (
    DEFAULT_NAMESPACE,
    OPTREE_NAMESPACES,
    PyTree,
    PyTreeNamespace,
)

if IS_JAX_INSTALLED:
    import jax.numpy as jnp  # type: ignore[import-not-found]

    JAX_ARRAY_TYPE: type = type(jnp.empty(0))


def tree_flatten(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
    namespace: PyTreeNamespace = DEFAULT_NAMESPACE,
) -> tuple[list[Any], PyTreeSpec]:
    """Flatten a pytree.

    Args:
        tree: The pytree to flatten.
        is_leaf: Optional function that returns True for subtrees that should be
            treated as leaves.
        namespace: The namespace that determines which types are internal nodes.

    Returns:
        The leaves of the tree and the tree definition.

    """
    _check_namespace(namespace)
    return optree.tree_flatten(tree, is_leaf=is_leaf, namespace=namespace)


def tree_leaves(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
    namespace: PyTreeNamespace = DEFAULT_NAMESPACE,
) -> list[Any]:
    """Get the leaves of a pytree.

    Args:
        tree: The pytree to flatten.
        is_leaf: Optional function that returns True for subtrees that should be
            treated as leaves.
        namespace: The namespace that determines which types are internal nodes.

    Returns:
        The leaves of the tree.

    """
    _check_namespace(namespace)
    return optree.tree_leaves(tree, is_leaf=is_leaf, namespace=namespace)


def tree_unflatten(
    treedef: PyTree | PyTreeSpec,
    leaves: Iterable[Any],
    namespace: PyTreeNamespace = DEFAULT_NAMESPACE,
) -> PyTree:
    """Reconstruct a pytree from the tree definition and the leaves.

    Args:
        treedef: A tree definition as returned by ``tree_flatten`` or a pytree with
            the desired structure.
        leaves: The leaves of the new tree.
        namespace: The namespace that determines which types are internal nodes.
            Only used if treedef is a pytree; a tree definition carries its own
            namespace.

    Returns:
        The reconstructed pytree.

    """
    _check_namespace(namespace)
    if not isinstance(treedef, PyTreeSpec):
        treedef = optree.tree_structure(treedef, namespace=namespace)
    return optree.tree_unflatten(treedef, leaves)


def tree_map(
    func: Callable[[Any], Any],
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
    namespace: PyTreeNamespace = DEFAULT_NAMESPACE,
) -> PyTree:
    """Apply a function to each leaf of a pytree.

    Args:
        func: The function that is applied to each leaf.
        tree: The pytree.
        is_leaf: Optional function that returns True for subtrees that should be
            treated as leaves.
        namespace: The namespace that determines which types are internal nodes.

    Returns:
        A pytree with the same structure as tree and transformed leaves.

    """
    _check_namespace(namespace)
    return optree.tree_map(func, tree, is_leaf=is_leaf, namespace=namespace)


def leaf_names(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
    namespace: PyTreeNamespace = DEFAULT_NAMESPACE,
    separator: str = "_",
) -> list[str]:
    """Get the path names of the leaves of a pytree.

    Args:
        tree: The pytree.
        is_leaf: Optional function that returns True for subtrees that should be
            treated as leaves.
        namespace: The namespace that determines which types are internal nodes.
        separator: The string used to join the entries of a path.

    Returns:
        One name per leaf, in the order of ``tree_leaves``.

    """
    names, _ = _flatten_with_names(tree, is_leaf, namespace, separator)
    return names


def _flatten_with_names(
    tree: PyTree,
    is_leaf: Callable[[PyTree], bool] | None,
    namespace: PyTreeNamespace,
    separator: str,
) -> tuple[list[str], list[Any]]:
    """Flatten a pytree and get the path names of its leaves in one pass."""
    _check_namespace(namespace)
    accessors, leaves, _ = optree.tree_flatten_with_accessor(
        tree, is_leaf=is_leaf, namespace=_get_names_namespace(namespace)
    )
    names = [
        separator.join(_entry_to_string(entry) for entry in accessor)
        for accessor in accessors
    ]
    return names, leaves


def _entry_to_string(entry: optree.PyTreeEntry) -> str:
    """Return the name of one accessor path entry.

    Namedtuple leaves are named by their field name instead of their position, so
    that leaf names stay aligned with how users refer to namedtuple parameters.
    """
    if isinstance(entry, optree.NamedTupleEntry):
        return entry.field
    return str(entry.entry)


def tree_equal(
    tree: PyTree,
    other: PyTree,
    is_leaf: Callable[[PyTree], bool] | None = None,
    namespace: PyTreeNamespace = DEFAULT_NAMESPACE,
    equality_checkers: dict[type, Callable[[Any, Any], bool]] | None = None,
) -> bool:
    """Check the equality between two trees.

    Two trees are considered equal if their leaf names and their leaves are equal.
    Leaves are compared with type-specific equality checkers. A checker normally
    returns a bool; checkers in the style of ``numpy.testing`` functions that raise
    on mismatch and return None are also supported and count as passing when they
    do not raise.

    Args:
        tree: The first pytree.
        other: The second pytree.
        is_leaf: Optional function that returns True for subtrees that should be
            treated as leaves.
        namespace: The namespace that determines which types are internal nodes.
        equality_checkers: Mapping from leaf types to functions that compare two
            leaves of that type. Extends and overrides the default checkers.

    Returns:
        True if the trees are equal.

    """
    equality_checkers = {**_get_equality_checkers(), **(equality_checkers or {})}

    first_names, first_leaves = _flatten_with_names(tree, is_leaf, namespace, "_")
    second_names, second_leaves = _flatten_with_names(other, is_leaf, namespace, "_")

    if first_names != second_names:
        return False

    for first, second in zip(first_leaves, second_leaves, strict=True):
        check_func = equality_checkers.get(type(first), lambda a, b: a == b)
        leaves_equal = check_func(first, second)
        if leaves_equal is not None and not bool(leaves_equal):
            return False

    return True


def _get_equality_checkers():
    """Return type-specific equality checkers for array and DataFrame leaves.

    These are used during pytree operations to compare leaves that don't
    support simple ``==`` equality (e.g. NumPy arrays, pandas objects).
    """
    equality_checkers = {}
    equality_checkers[np.ndarray] = lambda a, b: bool((a == b).all())
    equality_checkers[pd.Series] = lambda a, b: a.equals(b)
    equality_checkers[pd.DataFrame] = lambda a, b: a.equals(b)

    if IS_JAX_INSTALLED:
        equality_checkers[JAX_ARRAY_TYPE] = lambda a, b: bool((a == b).all())

    return equality_checkers


def _check_namespace(namespace: str) -> None:
    """Raise a ValueError if the namespace is not one of optimagic's namespaces."""
    valid = get_args(PyTreeNamespace)
    if namespace not in valid:
        raise ValueError(
            f"Invalid pytree namespace '{namespace}'. Must be one of: "
            f"{', '.join(valid)}."
        )


def _get_names_namespace(namespace: PyTreeNamespace) -> str:
    """Return the namespace whose flatten functions also return leaf path names.

    The default namespace registers no custom nodes and therefore needs no variant.
    """
    if namespace == DEFAULT_NAMESPACE:
        return namespace
    return f"{namespace}.names"


def _register_namespaces() -> contextlib.ExitStack:
    """Register flatten/unflatten functions and dict ordering for all namespaces.

    Each namespace in ``OPTREE_NAMESPACES`` is registered in two variants:

    1. The plain namespace, whose flatten functions return only the leaf values.
    2. A names variant (see ``_get_names_namespace``), whose flatten functions
       additionally return path names for the leaves.

    In all namespaces, dicts are flattened in insertion order instead of optree's
    default sorted order. optree only exposes this setting as a context manager
    that sets a process-wide flag. Entering and leaving it around each call would
    race between threads (e.g. with the threading batch evaluator), so the contexts
    are entered once here and never left.

    This function must only be called once, at import time.

    Returns:
        The exit stack holding the entered dict ordering contexts. It must stay
        referenced for the lifetime of the process; closing it would restore the
        sorted dict ordering.

    """
    for namespace in OPTREE_NAMESPACES:
        data_col = namespace.removeprefix(f"{DEFAULT_NAMESPACE}.")
        _register_namespace(namespace, data_col=data_col, with_names=False)
        _register_namespace(
            _get_names_namespace(namespace), data_col=data_col, with_names=True
        )

    stack = contextlib.ExitStack()
    for namespace in get_args(PyTreeNamespace):
        for ns in {namespace, _get_names_namespace(namespace)}:
            stack.enter_context(optree.dict_insertion_ordered(True, namespace=ns))
    return stack


def _register_namespace(namespace: str, data_col: str, with_names: bool) -> None:
    """Register flatten/unflatten functions for all supported types in a namespace."""
    optree.register_pytree_node(
        pd.DataFrame,
        partial(_flatten_df, data_col=data_col, with_names=with_names),
        partial(_unflatten_df, data_col=data_col),
        namespace=namespace,
    )

    optree.register_pytree_node(
        pd.Series,
        partial(_flatten_series, with_names=with_names),
        _unflatten_series,
        namespace=namespace,
    )

    optree.register_pytree_node(
        np.ndarray,
        partial(_flatten_ndarray, with_names=with_names),
        _unflatten_ndarray,
        namespace=namespace,
    )

    if IS_JAX_INSTALLED:
        optree.register_pytree_node(
            JAX_ARRAY_TYPE,
            partial(_flatten_jax_array, with_names=with_names),
            _unflatten_jax_array,
            namespace=namespace,
        )


def _flatten_df(df, data_col, with_names=False):
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
    entries = _get_df_names(df) if with_names else None
    return flat, aux_data, entries


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


def _flatten_series(series, with_names=False):
    """Flatten a series."""
    entries = list(series.index.map(_index_element_to_string)) if with_names else None
    return (
        series.tolist(),
        {"index": series.index, "name": series.name},
        entries,
    )


def _unflatten_series(aux_data, leaves):
    """Reconstruct a series."""
    return pd.Series(leaves, **aux_data)


def _flatten_ndarray(arr, with_names=False):
    """Flatten a numpy array."""
    entries = _array_element_names(arr) if with_names else None
    return arr.flatten().tolist(), arr.shape, entries


def _flatten_jax_array(arr, with_names=False):
    """Flatten a jax array."""
    entries = _array_element_names(arr) if with_names else None
    return arr.flatten().tolist(), arr.shape, entries


def _unflatten_jax_array(aux_data, leaves):
    """Reconstruct a jax array."""
    return jnp.array(leaves).reshape(aux_data)


def _unflatten_ndarray(aux_data, leaves):
    """Reconstruct a numpy array."""
    return np.array(leaves).reshape(aux_data)


def _get_df_names(df: pd.DataFrame) -> list[str]:
    """Get string names for dataframe leaf paths."""
    index_strings = list(df.index.map(_index_element_to_string))
    if "value" in df:
        out = index_strings
    else:
        out = ["_".join([loc, col]) for loc, col in product(index_strings, df.columns)]

    return out


def _index_element_to_string(element: Any) -> str:
    """Convert an index element to its string representation."""
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry) for entry in element]
        res_string = "_".join(as_strings)
    else:
        res_string = str(element)

    return res_string


def _array_element_names(arr: np.ndarray) -> list[str]:
    """Get string names for array like element leaf paths."""
    dim_names = [map(str, range(n)) for n in arr.shape]
    names = list(map("_".join, product(*dim_names)))
    return names


_DICT_INSERTION_ORDERING = _register_namespaces()

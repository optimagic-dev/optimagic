import itertools

import numpy as np
import pandas as pd


def _array_element_names(a):
    dim_names = [map(str, range(n)) for n in a.shape]
    names = list(map("_".join, itertools.product(*dim_names)))
    return names


def _index_element_to_string(element, separator="_"):
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry).replace("-", "_") for entry in element]
        res_string = separator.join(as_strings)
    else:
        res_string = str(element)
    return res_string


DEFAULT_PYTREE_REGISTRY = {
    list: {
        "flatten": lambda tree: (tree, None),
        "unflatten": lambda aux_data, children: children,
        "names": lambda tree: [f"{i}" for i in range(len(tree))],
    },
    dict: {
        "flatten": lambda tree: (list(tree.values()), list(tree)),
        "unflatten": lambda aux_data, children: dict(zip(aux_data, children)),
        "names": lambda tree: list(map(str, list(tree))),
    },
    tuple: {
        "flatten": lambda tree: (list(tree), None),
        "unflatten": lambda aux_data, children: tuple(children),
        "names": lambda tree: [f"{i}" for i in range(len(tree))],
    },
}


EXTENDED_PYTREE_REGISTRY = {
    np.ndarray: {
        "flatten": lambda tree: (tree.flatten().tolist(), tree.shape),
        "unflatten": lambda aux_data, children: np.array(children).reshape(aux_data),
        "names": _array_element_names,
    },
    pd.DataFrame: {
        "flatten": lambda tree: (tree["value"].tolist(), tree.drop(columns="value")),
        "unflatten": lambda aux_data, children: aux_data.assign(value=children),
        "names": lambda tree: list(tree.index.map(_index_element_to_string)),
    },
    pd.Series: {
        "flatten": lambda tree: (
            tree.tolist(),
            {"index": tree.index, "name": tree.name},
        ),
        "unflatten": lambda aux_data, children: pd.Series(children, **aux_data),
        "names": lambda tree: list(tree.index.map(_index_element_to_string)),
    },
    **DEFAULT_PYTREE_REGISTRY,
}


def tree_flatten(tree, is_leaf=None, registry=None):
    """Flatten a pytree.

    Args:
        tree: a pytree to flatten.
        is_leaf (callable): An optionally specified function that will be called at each
            flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict, None or "extended"): A pytree container registry that determines
            which types are considered container objects that should be flattened.
            `is_leaf` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.

    Returns:
        A pair where the first element is a list of leaf values and the second
        element is a treedef representing the structure of the flattened tree.

    """
    registry = _process_pytree_registry(registry)
    is_leaf = _process_is_leaf(is_leaf)

    flat = _tree_flatten(tree, is_leaf=is_leaf, registry=registry)
    dummy_flat = ["*"] * len(flat)
    treedef = tree_unflatten(tree, dummy_flat, is_leaf=is_leaf, registry=registry)
    return flat, treedef


def _tree_flatten(tree, is_leaf, registry):
    out = []
    tree_type = type(tree)

    if tree_type not in registry or is_leaf(tree):
        out.append(tree)
    else:
        subtrees, _ = registry[tree_type]["flatten"](tree)
        for subtree in subtrees:
            if type(subtree) in registry:
                out += _tree_flatten(subtree, is_leaf, registry)
            else:
                out.append(subtree)
    return out


def tree_unflatten(treedef, flat, is_leaf=None, registry=None):
    registry = _process_pytree_registry(registry)
    is_leaf = _process_is_leaf(is_leaf)
    return _tree_unflatten(treedef, flat, is_leaf=is_leaf, registry=registry)


def _tree_unflatten(treedef, flat, is_leaf, registry):
    flat = iter(flat)
    tree_type = type(treedef)

    if tree_type not in registry or is_leaf(treedef):
        return next(flat)
    else:
        items, info = registry[tree_type]["flatten"](treedef)
        unflattened_items = []
        for item in items:
            if type(item) in registry:
                unflattened_items.append(
                    _tree_unflatten(item, flat, is_leaf=is_leaf, registry=registry)
                )
            else:
                unflattened_items.append(next(flat))
        return registry[tree_type]["unflatten"](info, unflattened_items)


def tree_map(func, tree, is_leaf=None, registry=None):
    flat, treedef = tree_flatten(tree, is_leaf=is_leaf, registry=registry)
    modified = [func(i) for i in flat]
    return tree_unflatten(treedef, modified, is_leaf=is_leaf, registry=registry)


def tree_multimap(func, *trees, is_leaf=None, registry=None):
    flat_trees, treedefs = [], []
    for tree in trees:
        flat, treedef = tree_flatten(tree, is_leaf=is_leaf, registry=registry)
        flat_trees.append(flat)
        treedefs.append(treedef)

    for treedef in treedefs:
        if treedef != treedefs[0]:
            raise ValueError("All trees must have the same structure.")

    modified = [func(*item) for item in zip(*flat_trees)]

    out = tree_unflatten(treedefs[0], modified, is_leaf=is_leaf, registry=registry)
    return out


def leaf_names(tree, is_leaf=None, registry=None, separator="_"):
    registry = _process_pytree_registry(registry)
    is_leaf = _process_is_leaf(is_leaf)
    return _leaf_names(tree, is_leaf=is_leaf, registry=registry, separator=separator)


def _leaf_names(tree, is_leaf, registry, separator, prefix=None):
    out = []
    tree_type = type(tree)

    if tree_type not in registry or is_leaf(tree):
        out.append(prefix)
    else:
        subtrees, info = registry[tree_type]["flatten"](tree)
        names = registry[tree_type]["names"](tree)
        for name, subtree in zip(names, subtrees):
            if type(subtree) in registry:
                out += _leaf_names(
                    subtree,
                    is_leaf=is_leaf,
                    registry=registry,
                    separator=separator,
                    prefix=_add_prefix(prefix, name, separator),
                )
            else:
                out.append(_add_prefix(prefix, name, separator))
    return out


def _add_prefix(prefix, string, separator):
    if prefix not in (None, ""):
        out = separator.join([prefix, string])
    else:
        out = string
    return out


def _process_pytree_registry(registry):
    if registry is None:
        registry = DEFAULT_PYTREE_REGISTRY
    elif registry == "extended":
        registry = EXTENDED_PYTREE_REGISTRY

    return registry


def _process_is_leaf(is_leaf):
    if is_leaf is None:
        return lambda tree: False
    else:
        return is_leaf


def tree_equal(tree, other, is_leaf=None, registry=None):
    first_flat, first_treedef = tree_flatten(tree, is_leaf=is_leaf, registry=registry)
    second_flat, second_treedef = tree_flatten(
        other, is_leaf=is_leaf, registry=registry
    )

    equal = first_treedef == second_treedef
    for first, second in zip(first_flat, second_flat):
        if isinstance(first, np.ndarray):
            eq = (first == second).all()
        elif isinstance(first, (pd.DataFrame, pd.Series)):
            eq = first.equals(second)
        else:
            eq = first == second
        equal = equal and eq
    return equal


def tree_update(tree, other, is_leaf=None, registry=None):
    first_flat, first_treedef = tree_flatten(tree, is_leaf=is_leaf, registry=registry)
    first_names = leaf_names(tree, is_leaf=is_leaf, registry=registry)
    first_dict = dict(zip(first_names, first_flat))

    other_flat, _ = tree_flatten(other, is_leaf=is_leaf, registry=registry)
    other_names = leaf_names(other, is_leaf=is_leaf, registry=registry)
    other_dict = dict(zip(other_names, other_flat))

    combined = list({**first_dict, **other_dict}.values())

    out = tree_unflatten(first_treedef, combined, is_leaf=is_leaf, registry=registry)
    return out

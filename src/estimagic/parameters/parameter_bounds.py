import numpy as np
import pandas as pd
from estimagic.parameters.tree_registry import get_registry
from pybaum import tree_just_flatten as tree_leaves
from pybaum import tree_map
from pybaum import tree_update


def get_bounds(params, lower_bounds=None, upper_bounds=None):
    """Consolidate lower/upper bounds with bounds available in params.

    Updates bounds defined in params. If no bounds are available the entry is set to
    -np.inf for the lower bound and np.inf for the upper bound. If a bound is defined in
    params and lower_bounds or upper_bounds, the bound from lower_bounds or upper_bounds
    will be used.

    Args:
        params (pytree): The parameter pytree.
        lower_bounds (pytree): Must be a subtree of params.
        upper_bounds (pytree): Must be a subtree of params.

    Returns:
        np.ndarray: Consolidated and flattened lower_bounds.
        np.ndarray: Consolidated and flattened upper_bounds.

    """
    registry = get_registry(extended=True)
    n_params = len(tree_leaves(params, registry=registry))

    registry.pop(pd.DataFrame)
    bounds_tree = tree_map(
        lambda leaf: leaf if isinstance(leaf, pd.DataFrame) else np.nan,
        params,
        registry=registry,
    )

    lower_flat = _update_bounds_and_flatten(
        bounds_tree, lower_bounds, direction="lower_bound"
    )
    upper_flat = _update_bounds_and_flatten(
        bounds_tree, upper_bounds, direction="upper_bound"
    )

    if len(lower_flat) != n_params:
        raise ValueError("lower_bounds do not match dimension of params.")
    if len(upper_flat) != n_params:
        raise ValueError("upper_bounds do not match dimension of params.")

    lower_flat = np.nan_to_num(lower_flat, nan=-np.inf)
    upper_flat = np.nan_to_num(upper_flat, nan=np.inf)

    return lower_flat, upper_flat


def _update_bounds_and_flatten(bounds_tree, bounds, direction):
    registry = get_registry(extended=True, data_col=direction)
    if bounds is not None:
        bounds_tree = tree_update(bounds_tree, bounds)
    bounds_flat = tree_leaves(bounds_tree, registry=registry)
    bounds_flat = np.array(bounds_flat, dtype=np.float64)
    return bounds_flat

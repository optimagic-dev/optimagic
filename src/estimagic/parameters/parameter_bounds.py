import numpy as np
from estimagic.parameters.tree_registry import get_registry
from pybaum import tree_just_flatten as tree_leaves
from pybaum import tree_map
from pybaum import tree_update


def get_bounds(
    params,
    lower_bounds=None,
    upper_bounds=None,
    soft_lower_bounds=None,
    soft_upper_bounds=None,
    registry=None,
    add_soft_bounds=False,
):
    """Consolidate lower/upper bounds with bounds available in params.

    Updates bounds defined in params. If no bounds are available the entry is set to
    -np.inf for the lower bound and np.inf for the upper bound. If a bound is defined in
    params and lower_bounds or upper_bounds, the bound from lower_bounds or upper_bounds
    will be used.

    Args:
        params (pytree): The parameter pytree.
        lower_bounds (pytree): Must be a subtree of params.
        upper_bounds (pytree): Must be a subtree of params.
        registry (dict): pybaum registry.

    Returns:
        np.ndarray: Consolidated and flattened lower_bounds.
        np.ndarray: Consolidated and flattened upper_bounds.

    """
    fast_path = _is_fast_path(
        params=params,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        add_soft_bounds=add_soft_bounds,
    )
    if fast_path:
        return _get_fast_path_bounds(
            params=params,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

    registry = get_registry(extended=True) if registry is None else registry
    n_params = len(tree_leaves(params, registry=registry))

    # Fill leaves with np.nan. If params contains a data frame with bounds column, that
    # column is not overwritten.
    bounds_tree = tree_map(lambda leaf: np.nan, params, registry=registry)

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

    lower_flat[np.isnan(lower_flat)] = -np.inf
    upper_flat[np.isnan(upper_flat)] = np.inf

    if add_soft_bounds:
        lower_flat_soft = _update_bounds_and_flatten(
            bounds_tree, soft_lower_bounds, direction="soft_lower_bound"
        )
        lower_flat_soft[np.isnan(lower_flat_soft)] = -np.inf
        lower_flat = np.maximum(lower_flat, lower_flat_soft)

        upper_flat_soft = _update_bounds_and_flatten(
            bounds_tree, soft_upper_bounds, direction="soft_upper_bound"
        )
        upper_flat_soft[np.isnan(upper_flat_soft)] = np.inf
        upper_flat = np.minimum(upper_flat, upper_flat_soft)

    if (lower_flat > upper_flat).any():
        msg = "Invalid bounds. Some lower bounds are larger than upper bounds."
        raise ValueError(msg)

    return lower_flat, upper_flat


def _update_bounds_and_flatten(bounds_tree, bounds, direction):
    registry = get_registry(extended=True, data_col=direction)
    if bounds is not None:
        bounds_tree = tree_update(bounds_tree, bounds)
    bounds_flat = tree_leaves(bounds_tree, registry=registry)
    bounds_flat = np.array(bounds_flat, dtype=np.float64)
    return bounds_flat


def _is_fast_path(params, lower_bounds, upper_bounds, add_soft_bounds):
    out = True
    if add_soft_bounds:
        out = False

    if not _is_1d_array(params):
        out = False

    for bound in lower_bounds, upper_bounds:
        if not (_is_1d_array(bound) or bound is None):
            out = False
    return out


def _is_1d_array(candidate):
    return isinstance(candidate, np.ndarray) and candidate.ndim == 1


def _get_fast_path_bounds(params, lower_bounds, upper_bounds):
    if lower_bounds is None:
        # faster than np.full
        lower_bounds = np.array([-np.inf] * len(params))
    else:
        lower_bounds = lower_bounds.astype(float)

    if upper_bounds is None:
        # faster than np.full
        upper_bounds = np.array([np.inf] * len(params))
    else:
        upper_bounds = upper_bounds.astype(float)

    if (lower_bounds > upper_bounds).any():
        msg = "Invalid bounds. Some lower bounds are larger than upper bounds."
        raise ValueError(msg)

    return lower_bounds, upper_bounds

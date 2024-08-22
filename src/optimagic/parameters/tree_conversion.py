from typing import Callable, NamedTuple

import numpy as np
from pybaum import leaf_names, tree_flatten, tree_just_flatten, tree_unflatten

from optimagic.exceptions import InvalidFunctionError
from optimagic.parameters.block_trees import block_tree_to_matrix
from optimagic.parameters.bounds import get_internal_bounds
from optimagic.parameters.tree_registry import get_registry
from optimagic.typing import AggregationLevel


def get_tree_converter(
    params,
    bounds,
    func_eval,
    solver_type,
    derivative_eval=None,
    add_soft_bounds=False,
):
    """Get flatten and unflatten functions for criterion and its derivative.

    The function creates a converter with methods to convert parameters, derivatives
    and the output of the criterion function between the user provided pytree structure
    and flat representations.

    The main motivation for bundling all of this together (as opposed to handling
    parameters, derivatives and function outputs separately) is that the derivative
    conversion needs to know about the structure of params and the criterion output.

    Args:
        params (pytree): The user provided parameters.
        lower_bounds (pytree): The user provided lower_bounds
        upper_bounds (pytree): The user provided upper bounds
        solver_type: Used to determine how derivative output has to be
            transformed for the optimizer.
        derivative_eval (dict, pytree or None): Evaluation of the derivative of
            func at params. Used for consistency checks.
        soft_lower_bounds (pytree): As lower_bounds
        soft_upper_bounds (pytree): As upper_bounds
        add_soft_bounds (bool): Whether soft bounds should be added to the flat_params

    Returns:
        TreeConverter: NamedTuple with flatten and unflatten methods.
        FlatParams: NamedTuple of 1d arrays with flattened bounds and param names.

    """
    _registry = get_registry(extended=True)
    _params_vec, _params_treedef = tree_flatten(params, registry=_registry)
    _params_vec = np.array(_params_vec).astype(float)
    _lower, _upper = get_internal_bounds(
        params=params,
        bounds=bounds,
        registry=_registry,
    )

    if add_soft_bounds:
        _soft_lower, _soft_upper = get_internal_bounds(
            params=params,
            bounds=bounds,
            registry=_registry,
            add_soft_bounds=add_soft_bounds,
        )
    else:
        _soft_lower, _soft_upper = None, None

    _param_names = leaf_names(params, registry=_registry)

    flat_params = FlatParams(
        values=_params_vec,
        lower_bounds=_lower,
        upper_bounds=_upper,
        names=_param_names,
        soft_lower_bounds=_soft_lower,
        soft_upper_bounds=_soft_upper,
    )

    _params_flatten = _get_params_flatten(registry=_registry)
    _params_unflatten = _get_params_unflatten(
        registry=_registry, treedef=_params_treedef
    )

    _derivative_flatten = _get_derivative_flatten(
        registry=_registry,
        solver_type=solver_type,
        params=params,
        func_eval=func_eval,
        derivative_eval=derivative_eval,
    )

    converter = TreeConverter(
        params_flatten=_params_flatten,
        params_unflatten=_params_unflatten,
        derivative_flatten=_derivative_flatten,
    )

    return converter, flat_params


def _get_params_flatten(registry):
    def params_flatten(params):
        return np.array(tree_just_flatten(params, registry=registry)).astype(float)

    return params_flatten


def _get_params_unflatten(registry, treedef):
    def params_unflatten(x):
        return tree_unflatten(treedef=treedef, leaves=list(x), registry=registry)

    return params_unflatten


def _get_best_key_and_aggregator(needed_key, available_keys):
    if needed_key in available_keys:
        key = needed_key
        if needed_key == "value":
            aggregate = lambda x: float(x[0])
        else:
            aggregate = lambda x: np.array(x).astype(float)
    elif needed_key == "contributions" and "root_contributions" in available_keys:
        key = "root_contributions"
        aggregate = lambda x: np.array(x).astype(float) ** 2
    elif needed_key == "value" and "contributions" in available_keys:
        key = "contributions"
        aggregate = lambda x: float(np.sum(x))
    elif needed_key == "value" and "root_contributions" in available_keys:
        key = "root_contributions"
        aggregate = lambda x: float((np.array(x) ** 2).sum())
    else:
        msg = (
            "The optimizer you requested requires a criterion function that returns "
            f"a dictionary with the entry '{needed_key}'. Your function returns a "
            f"dictionary that only contains the entries {available_keys}."
        )
        raise InvalidFunctionError(msg)

    return key, aggregate


def _get_derivative_flatten(registry, solver_type, params, func_eval, derivative_eval):
    # gradient case
    if solver_type == AggregationLevel.SCALAR:

        def derivative_flatten(derivative_eval):
            flat = np.array(
                tree_just_flatten(derivative_eval, registry=registry)
            ).astype(float)
            return flat

    # jacobian case
    else:

        def derivative_flatten(derivative_eval):
            flat = block_tree_to_matrix(
                derivative_eval,
                outer_tree=func_eval,
                inner_tree=params,
            )
            return flat

    if derivative_eval is not None:
        try:
            derivative_flatten(derivative_eval)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            msg = "The output of derivative and criterion cannot be aligned."
            raise InvalidFunctionError(msg) from e

    return derivative_flatten


class TreeConverter(NamedTuple):
    params_flatten: Callable
    params_unflatten: Callable
    derivative_flatten: Callable


class FlatParams(NamedTuple):
    values: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    soft_lower_bounds: np.ndarray | None = None
    soft_upper_bounds: np.ndarray | None = None
    names: list | None = None
    free_mask: np.ndarray | None = None

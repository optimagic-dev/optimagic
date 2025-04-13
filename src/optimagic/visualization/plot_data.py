import warnings
from functools import partial

import numpy as np
import pandas as pd
from pybaum import tree_just_flatten

from optimagic import deprecations
from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.deprecations import replace_and_warn_about_deprecated_bounds
from optimagic.optimization.fun_value import (
    convert_fun_output_to_function_value,
    enforce_return_type,
)
from optimagic.parameters.bounds import pre_process_bounds
from optimagic.parameters.conversion import get_converter
from optimagic.parameters.tree_registry import get_registry
from optimagic.shared.process_user_function import infer_aggregation_level
from optimagic.typing import AggregationLevel


def evaluate_func(params, func, func_kwargs):  # type: ignore
    """Evaluate a user-defined function, handling deprecated dictionary output.

    Args:
        params: Input parameters for the function.
        func: The user-defined objective function.
        func_kwargs: Optional dictionary of keyword arguments to pass to the function.

    Returns:
        A tuple of (possibly wrapped) function and its evaluated output.

    """
    if func_kwargs:
        func = partial(func, **func_kwargs)

    func_eval = func(params)

    if deprecations.is_dict_output(func_eval):
        warnings.warn(
            "Functions that return dictionaries are deprecated and will "
            "raise an error in future versions.",
            FutureWarning,
        )
        func_eval = deprecations.convert_dict_to_function_value(func_eval)
        func = deprecations.replace_dict_output(func)

    problem_type = (
        deprecations.infer_problem_type_from_dict_output(func_eval)
        if deprecations.is_dict_output(func_eval)
        else infer_aggregation_level(func)
    )
    func_eval = convert_fun_output_to_function_value(func_eval, problem_type)
    func = enforce_return_type(problem_type)(func)
    return func, func_eval


def process_bounds(bounds, lower_bounds, upper_bounds):  # type: ignore
    """Process parameter bounds, replacing deprecated formats if necessary.

    Args:
        bounds: Bound object or structure.
        lower_bounds: Deprecated lower bounds.
        upper_bounds: Deprecated upper bounds.

    Returns:
        Processed and validated bounds.

    """
    bounds = replace_and_warn_about_deprecated_bounds(
        bounds=bounds, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    return pre_process_bounds(bounds)


def select_parameter_indices(converter, selector, n_params):  # type: ignore
    """Select parameter indices using a selector function, or select all by default.

    Args:
        converter: Parameter converter.
        selector: Callable to select specific parameters.
        n_params: Total number of parameters.

    Returns:
        Array of selected parameter indices.

    """
    if selector is None:
        return np.arange(n_params, dtype=int)

    helper = converter.params_from_internal(np.arange(n_params))
    registry = get_registry(extended=True)
    return np.array(tree_just_flatten(selector(helper), registry=registry), dtype=int)


def generate_grid_data(internal_params, selected, n_gridpoints):  # type: ignore
    """Generate a grid of parameter values based on selection.

    Args:
        internal_params: Internal representation of parameters.
        selected: List of indices for parameters to vary.
        n_gridpoints: Number of values to generate per parameter.

    Returns:
        DataFrame containing the grid of parameter values.

    """
    metadata = {
        name: (
            np.linspace(
                internal_params.lower_bounds[pos],
                internal_params.upper_bounds[pos],
                n_gridpoints,
            )
            if pos in selected
            else internal_params.values[pos]
        )
        for pos, name in enumerate(internal_params.names)
    }
    return pd.DataFrame(metadata)


def generate_internal_params(params, bounds, func_eval):  # type: ignore
    """Construct internal parameter conversion from given inputs.

    Args:
        params: Original parameter values.
        bounds: Bound constraints.
        func_eval: Evaluated function value for structure.

    Returns:
        A converter object for internal parameters.

    """
    return get_converter(
        params=params,
        constraints=None,
        bounds=bounds,
        func_eval=func_eval,
        solver_type="value",
    )


def evaluate_function_values(func, evaluation_points, batch_evaluator, n_cores):  # type: ignore
    """Evaluate function at multiple points using a batch evaluation strategy.

    Args:
        func: The function to evaluate.
        evaluation_points: List of input parameter values.
        batch_evaluator: Function or callable that evaluates in batch.
        n_cores: Number of CPU cores for parallelism.

    Returns:
        List of evaluated function values or NaNs for failed evaluations.

    """
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    results = batch_evaluator(
        func=func,
        arguments=evaluation_points,
        error_handling="continue",
        n_cores=n_cores,
    )
    return [
        float("nan")
        if isinstance(val, str)
        else val.internal_value(AggregationLevel.SCALAR)
        for val in results
    ]


def generate_eval_points(grid, params, param_names, fixed_vars, converter, projection):  # type: ignore
    """Generate evaluation points based on a grid of selected parameters and fixed
    variables.

    This function supports two modes:
    - If `projection` is not "slice",
    a full 2D meshgrid of points is generated for the two selected parameters.
    - If `projection` is "slice",
    only the selected parameters are varied individually.

    Args:
        grid: DataFrame of generated parameter values.
        params: Internal parameter structure.
        param_names: Names of parameters to vary.
        fixed_vars: Dictionary of fixed parameter values.
        converter: Converter object to map to internal parameter format.
        projection: Projection mode ("contour", "3d", or "slice").

    Returns:
        If projection is not "slice":
            Tuple of meshgrid arrays (X, Y) and list of evaluation points.
        If projection is "slice":
            Tuple of selected input values (X) and list of evaluation points.

    """
    evaluation_points = []

    if projection != "slice":
        x_vals = grid[param_names[0]].to_numpy()
        y_vals = grid[param_names[1]].to_numpy()
        X, Y = np.meshgrid(x_vals, y_vals)

        for a, b in zip(X.ravel(), Y.ravel(), strict=False):
            point_dict = {param_names[0]: a, param_names[1]: b, **fixed_vars}
            internal_values = np.array(list(point_dict.values()))
            evaluation_points.append(converter.params_from_internal(internal_values))

        return X, Y, evaluation_points

    else:
        X = grid[param_names].to_numpy()
        for param_value in X:
            point_dict = (
                {**fixed_vars, param_names: param_value}
                if isinstance(param_names, str)
                else {
                    **fixed_vars,
                    **dict(zip(param_names, param_value, strict=False)),
                }
            )

            internal_values = np.array(
                [
                    point_dict.get(name, params.values[params.names.index(name)])
                    for name in params.names
                ]
            )
            evaluation_points.append(converter.params_from_internal(internal_values))
        return X, evaluation_points

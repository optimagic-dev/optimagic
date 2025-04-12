import warnings
from functools import partial

import numpy as np
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from pybaum import tree_just_flatten

from optimagic import deprecations
from optimagic.batch_evaluators import process_batch_evaluator
from optimagic.config import DEFAULT_N_CORES, PLOTLY_TEMPLATE
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
from optimagic.visualization.plotting_utilities import combine_plots, get_layout_kwargs


def evaluate_func(params, func, func_kwargs):
    if func_kwargs:
        func = partial(func, **func_kwargs)
    func_eval = func(params)

    if deprecations.is_dict_output(func_eval):
        warnings.warn(
            "Functions that return dictionaries are deprecated and will"
            " raise an error in future versions.",
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


def process_bounds(bounds, lower_bounds, upper_bounds):
    bounds = replace_and_warn_about_deprecated_bounds(
        bounds=bounds, lower_bounds=lower_bounds, upper_bounds=upper_bounds
    )
    return pre_process_bounds(bounds)


def generate_internal_params(params, bounds, func_eval):
    return get_converter(
        params=params,
        constraints=None,
        bounds=bounds,
        func_eval=func_eval,
        solver_type="value",
    )


def select_parameter_indices(converter, selector, n_params):
    if selector is None:
        return np.arange(n_params, dtype=int)
    helper = converter.params_from_internal(np.arange(n_params))
    registry = get_registry(extended=True)
    return np.array(tree_just_flatten(selector(helper), registry=registry), dtype=int)


def generate_grid_data(internal_params, selected, n_gridpoints):
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


def generate_evaluation_points(
    grid_data, internal_params, selected_names, fixed_vars, converter, projection
):
    evaluation_points = []
    if projection != "slice":
        X, Y = np.meshgrid(
            grid_data[selected_names[0]].values, grid_data[selected_names[1]].values
        )
        for a, b in zip(X.ravel(), Y.ravel(), strict=False):
            point_dict = {selected_names[0]: a, selected_names[1]: b, **fixed_vars}
            internal_values = np.array(list(point_dict.values()))
            evaluation_points.append(converter.params_from_internal(internal_values))
        return X, Y, evaluation_points
    else:
        X = grid_data[selected_names].values
        for param_value in X:
            point_dict = {**fixed_vars, selected_names: param_value}
            internal_values = np.array(
                [
                    point_dict.get(
                        name, internal_params.values[internal_params.names.index(name)]
                    )
                    for name in internal_params.names
                ]
            )
            evaluation_points.append(converter.params_from_internal(internal_values))
        return X, evaluation_points


def evaluate_function_values(func, evaluation_points, batch_evaluator, n_cores):
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    results = batch_evaluator(
        func=func,
        arguments=evaluation_points,
        error_handling="continue",
        n_cores=n_cores,
    )
    return [
        np.nan if isinstance(val, str) else val.internal_value(AggregationLevel.SCALAR)
        for val in results
    ]


def _plot_slice(
    selected,
    internal_params,
    converter,
    grid_data,
    func,
    func_eval,
    batch_evaluator,
    n_cores,
    param_names,
    expand_yrange,
    color,
    template,
    title,
    plots_per_row,
    share_y,
    share_x,
    return_dict,
    make_subplot_kwargs,
):
    plots_dict = {}
    title_kwargs = {"text": title} if title else None
    layout_kwargs = get_layout_kwargs(None, None, title_kwargs, template, False)

    for idx in selected:
        param_name = internal_params.names[idx]
        fixed_vars = {
            name: internal_params.values[internal_params.names.index(name)]
            for name in internal_params.names
            if name != param_name
        }
        X, evaluation_points = generate_evaluation_points(
            grid_data, internal_params, param_name, fixed_vars, converter, "slice"
        )
        func_values = evaluate_function_values(
            func, evaluation_points, batch_evaluator, n_cores
        )

        y_min, y_max = np.min(func_values), np.max(func_values)
        y_range = y_max - y_min
        yaxis_range = [y_min - y_range * expand_yrange, y_max + y_range * expand_yrange]

        display_name = (
            param_names.get(param_name, param_name) if param_names else param_name
        )
        fig = px.line(
            x=grid_data[param_name].values,
            y=func_values,
            color_discrete_sequence=[color],
        )
        fig.add_trace(
            go.Scatter(
                x=[internal_params.values[idx]],
                y=[func_eval.internal_value(AggregationLevel.SCALAR)],
                marker={"color": color},
            )
        )
        fig.update_layout(**layout_kwargs)
        fig.update_xaxes(title={"text": display_name})
        fig.update_yaxes(
            title={"text": "Function Value"}, range=yaxis_range if share_y else None
        )
        plots_dict[display_name] = fig

    if return_dict:
        return plots_dict
    return combine_plots(
        plots=list(plots_dict.values()),
        plots_per_row=plots_per_row,
        sharex=share_x,
        sharey=share_y,
        share_yrange_all=share_y,
        share_xrange_all=share_x,
        expand_yrange=expand_yrange,
        make_subplot_kwargs=make_subplot_kwargs,
        showlegend=False,
        template=template,
        clean_legend=True,
        layout_kwargs=layout_kwargs,
        legend_kwargs={},
        title_kwargs=title_kwargs,
    )


def _plot_pairwise(
    selected,
    internal_params,
    converter,
    grid_data,
    func,
    batch_evaluator,
    n_cores,
    projection,
    title,
    template,
    n_gridpoints,
):
    selected_param_names = [internal_params.names[i] for i in selected]
    specs = [
        [
            {"type": "scene"} if projection == "3d" else {"type": "xy"}
            for _ in range(len(selected))
        ]
        for _ in range(len(selected))
    ]
    subplot_titles = [
        f"{selected_param_names[j]} vs {selected_param_names[i]}" if i != j else ""
        for i in range(len(selected))
        for j in range(len(selected))
    ]
    fig = make_subplots(
        rows=len(selected),
        cols=len(selected),
        specs=specs,
        subplot_titles=subplot_titles,
    )

    for i, name_i in enumerate(selected_param_names):
        for j, name_j in enumerate(selected_param_names):
            selected_names = [name_j, name_i]
            fixed_vars = {
                name: grid_data.iloc[0][name]
                for name in internal_params.names
                if name not in selected_names
            }
            X, Y, evaluation_points = generate_evaluation_points(
                grid_data,
                internal_params,
                selected_names,
                fixed_vars,
                converter,
                projection,
            )
            func_values = evaluate_function_values(
                func, evaluation_points, batch_evaluator, n_cores
            )
            Z = np.reshape(func_values, (n_gridpoints, n_gridpoints))
            trace = (
                go.Surface(z=Z, x=X, y=Y, showscale=False, colorscale="Viridis")
                if projection == "3d"
                else go.Contour(
                    z=Z,
                    x=X[0],
                    y=Y[:, 0],
                    colorscale="Viridis",
                    contours_coloring="heatmap",
                    line_smoothing=0.85,
                )
            )
            fig.add_trace(trace, row=i + 1, col=j + 1)

    fig.update_layout(
        title=title or f"Function plot - ({projection})", template=template
    )
    return fig


def slice_plot(
    func,
    params,
    bounds=None,
    func_kwargs=None,
    selector=None,
    n_cores=DEFAULT_N_CORES,
    n_gridpoints=20,
    plots_per_row=2,
    param_names=None,
    share_y=True,
    expand_yrange=0.02,
    share_x=False,
    color="#497ea7",
    template=PLOTLY_TEMPLATE,
    title=None,
    return_dict=False,
    make_subplot_kwargs=None,
    batch_evaluator="joblib",
    projection="slice",
    # deprecated
    lower_bounds=None,
    upper_bounds=None,
):
    # Preprocess function and bounds
    func, func_eval = evaluate_func(params, func, func_kwargs)
    bounds = process_bounds(bounds, lower_bounds, upper_bounds)

    # Generate internal parameter representation
    converter, internal_params = generate_internal_params(params, bounds, func_eval)
    n_params = len(internal_params.values)

    # Select parameters for plotting
    selected = select_parameter_indices(converter, selector, n_params)
    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")
    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    # Create grid data
    grid_data = generate_grid_data(internal_params, selected, n_gridpoints)

    if projection == "slice":
        return _plot_slice(
            selected,
            internal_params,
            converter,
            grid_data,
            func,
            func_eval,
            batch_evaluator,
            n_cores,
            param_names,
            expand_yrange,
            color,
            template,
            title,
            plots_per_row,
            share_y,
            share_x,
            return_dict,
            make_subplot_kwargs,
        )
    if projection in {"3d", "contour"}:
        return _plot_pairwise(
            selected,
            internal_params,
            converter,
            grid_data,
            func,
            batch_evaluator,
            n_cores,
            projection,
            title,
            template,
            n_gridpoints,
        )

    return None

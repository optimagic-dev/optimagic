import warnings
from copy import deepcopy
from enum import Enum, auto
from functools import partial

import numpy as np
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


def slice_plot_3d(
    func,
    params,
    bounds=None,
    func_kwargs=None,
    selector=None,
    n_gridpoints=20,
    projection="slice",
    make_subplot_kwargs=None,
    layout_kwargs=None,
    plot_kwargs=None,
    param_names=None,
    expand_yrange=0.02,
    batch_evaluator="joblib",
    n_cores=DEFAULT_N_CORES,
    return_dict=False,
    lower_bounds=None,
    upper_bounds=None,
):
    # Projection evaluation
    projection = Projection.from_value(projection)
    if projection.is_multiple():
        template = "plotly"
    else:
        template = PLOTLY_TEMPLATE

    plot_kwargs = plot_kwargs if plot_kwargs is not None else {}
    make_subplot_kwargs = make_subplot_kwargs if make_subplot_kwargs is not None \
                                                else {}
    layout_kwargs = layout_kwargs if layout_kwargs is not None else {}

    # Preprocess objective function and bounds
    func, func_eval = evaluate_func(params, func, func_kwargs)
    bounds = process_bounds(bounds, lower_bounds, upper_bounds)

    # Generate internal parameters and converter
    converter, internal_params = get_converter(
        params=params,
        constraints=[],
        bounds=bounds,
        func_eval=func_eval,
        solver_type="value",
    )

    # Subset of parameters which is selected
    selected = select_parameter_indices(converter, internal_params, selector)
    selected_size = len(selected)

    # Validate enough parameters for "3d" and "contour" projections
    if projection.is_multiple() and selected_size < 2:
        raise ValueError(
            f"{projection!r} projection requires at least two parameters."
            f"Got {selected_size}. Please revise the `selector`."
        )

    # Create param data
    param_data = generate_param_data(internal_params, selected, n_gridpoints)

    # Evaluate kwargs
    make_subplot_kwargs, layout_kwargs, plot_kwargs = evaluate_kwargs(
        projection,
        selected_size,
        make_subplot_kwargs,
        layout_kwargs,
        plot_kwargs,
        return_dict=return_dict,
        template=template,
    )

    plots = {}
    if projection.is_single():
        cols = make_subplot_kwargs.get("cols")
        for index, pos in enumerate(selected):
            param_name = internal_params.names[pos]
            param_value = internal_params.values[pos]
            fig = plot_slice_util(param_data, param_name, param_value,
                                  param_names, internal_params, func,
                                  func_eval, converter,
                                  batch_evaluator, n_cores, expand_yrange,
                                  plot_kwargs, make_subplot_kwargs)

            row = index // cols
            col = index % cols
            plots[(row, col)] = fig
    else:
        single_plot_flag = selected_size == 2
        for index_x, pos_x in enumerate(selected):
            for index_y, pos_y in enumerate(selected):
                if pos_x == pos_y and not single_plot_flag:
                    param_name = internal_params.names[pos_x]
                    param_value = internal_params.values[pos_x]
                    fig = plot_slice_util(param_data, param_name, param_value,
                                          param_names, internal_params, func,
                                          func_eval,
                                          converter, batch_evaluator, n_cores,
                                          expand_yrange,
                                          plot_kwargs, make_subplot_kwargs)
                else:
                    x_param_name = internal_params.names[pos_x]
                    y_param_name = internal_params.names[pos_y]

                    display_names = {
                        "x": param_names.get(x_param_name, x_param_name)
                        if param_names
                        else x_param_name,
                        "y": param_names.get(y_param_name, y_param_name)
                        if param_names
                        else y_param_name,
                    }

                    x = param_data[x_param_name]
                    y = param_data[y_param_name]
                    z = evaluate_function_values(
                        param_data,
                        internal_params,
                        [x_param_name, y_param_name],
                        func,
                        converter,
                        batch_evaluator,
                        n_cores,
                        projection=projection,
                    )

                    x, y = np.meshgrid(x, y)
                    z = np.reshape(z, (n_gridpoints, n_gridpoints))

                    if projection == Projection.SURFACE:
                        fig = plot_surface(
                            x, y, z, display_names=display_names,
                            plot_kwargs=plot_kwargs
                        )
                    else:
                        fig = plot_contour(
                            x, y, z, display_names=display_names,
                            plot_kwargs=plot_kwargs
                        )

                if single_plot_flag:
                    plots[(0, 0)] = fig
                    break
                plots[(index_x, index_y)] = fig
            if single_plot_flag:
                break

    return (
        plots
        if return_dict
        else combine_plots(plots, make_subplot_kwargs, layout_kwargs, expand_yrange)
    )


def plot_slice(
    x,
    y,
    y_range=None,
    point=None,
    display_name=None,
    plot_kwargs=None,
    make_subplot_kwargs=None,
):
    fig = px.line(x=x, y=y, **plot_kwargs["line_plot"])

    fig.add_trace(go.Scatter(x=point["x"], y=point["y"], **plot_kwargs["scatter_plot"]))

    layout_kwargs = dict(
        title=f"test: {display_name}",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(title={"text": display_name})
    fig.update_yaxes(
        title={"text": "Function Value"},
        range=y_range
        if "shared_yaxes" in make_subplot_kwargs
           and make_subplot_kwargs["shared_yaxes"]
        else None,
    )
    return fig


def plot_surface(x, y, z, display_names=None, plot_kwargs=None):
    trace = go.Surface(z=z, x=x, y=y, **plot_kwargs["surface_plot"])
    layout_kwargs = dict(
        title=f"{display_names['x']} vs {display_names['y']}",
        template=PLOTLY_TEMPLATE,
        autosize=False,
        scene_camera_eye=dict(x=0, y=0, z=-0.64),
    )

    fig = go.Figure(data=[trace], layout=layout_kwargs)

    # fig.update_layout(**layout_kwargs)
    # fig.update_xaxes(mirror=True)
    return fig


def plot_contour(x, y, z, display_names=None, plot_kwargs=None):
    trace = go.Contour(z=z, x=x[0], y=y[:, 0], **plot_kwargs["contour_plot"])
    fig = go.Figure(data=[trace])

    layout_kwargs = dict(
        title=f"{display_names['x']} vs {display_names['y']}",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(**layout_kwargs)
    return fig


def plot_slice_util(param_data, param_name, param_value,
                    param_names, internal_params, func, func_eval,
                    converter,
                    batch_evaluator, n_cores, expand_yrange,
                    plot_kwargs, make_subplot_kwargs):
    display_name = (
        param_names.get(param_name, param_name) if param_names else param_name
    )

    x = param_data[param_name].tolist()
    y = evaluate_function_values(
        param_data,
        internal_params,
        param_name,
        func,
        converter,
        batch_evaluator,
        n_cores,
    )

    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    yaxis_range = [
        y_min - y_range * expand_yrange,
        y_max + y_range * expand_yrange,
        ]

    fig = plot_slice(
        x,
        y,
        point={
            "x": [param_value],
            "y": [func_eval.internal_value(AggregationLevel.SCALAR)],
        },
        y_range=yaxis_range,
        display_name=display_name,
        plot_kwargs=plot_kwargs,
        make_subplot_kwargs=make_subplot_kwargs,
    )

    return fig


# Plot utils
class Projection(Enum):
    SLICE = auto()
    CONTOUR = auto()
    SURFACE = auto()

    @classmethod
    def from_value(cls, value):
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except ValueError as err:
                raise ValueError(f"Invalid projection : '{value}'") from err
        elif isinstance(value, cls):
            return value
        else:
            raise TypeError(f"Expected str or Projection(Enum), got {type(value)}")

    def is_single(self):
        return self == Projection.SLICE

    def is_multiple(self):
        return self in (Projection.CONTOUR, Projection.SURFACE)

    def __str__(self):
        return f"Projection({self.name})"


def update_nested_dict(default, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and k in default and isinstance(default[k], dict):
            # If both are dicts, recurse
            default[k] = update_nested_dict(default[k], v)
        else:
            # Otherwise, replace value
            default[k] = v
    return default


def get_layout_kwargs(
        layout_kwargs,
        rows,
        cols,
        return_dict=False,
        template=PLOTLY_TEMPLATE,
        single_plot=False,
):
    if return_dict or single_plot:
        width = 500
        height = 500
    else:
        width = 350 * cols
        height = 350 * rows
    layout_defaults = {
        "width": width,
        "height": height,
        "template": template,
        "xaxis_showgrid": False,
        "yaxis_showgrid": False,
        "title": {},
        "legend": {},
    }

    if layout_kwargs:
        layout_defaults.update(layout_kwargs)

    return layout_defaults


def get_make_subplot_kwargs(make_subplot_kwargs, rows, cols, projection=None,
                            single_plot=False):
    make_subplot_defaults = {
        "rows": rows,
        "cols": cols,
        "start_cell": "top-left",
        "print_grid": False,
        "horizontal_spacing": 1 / (cols * 5),
        "vertical_spacing": (1 / max(rows - 1, 1)) / 5,
    }
    if make_subplot_kwargs:
        make_subplot_defaults.update(make_subplot_kwargs)
    if projection and projection.is_multiple():
        make_subplot_defaults["specs"] = [
            [
                {"type": "xy"} if row == col and not single_plot else
                {"type": "scene"} if projection == Projection.SURFACE else
                {"type": "xy"}
                for col in range(cols)
            ]
            for row in range(rows)
        ]
        # make_subplot_defaults["subplot_titles"] = [
        #     f"{selected_param_names[j]} vs {selected_param_names[i]}"
        #     if i != j else ""
        #     for i in range(cols)
        #     for j in range(rows)
        # ]
    return make_subplot_defaults


def get_plot_kwargs(projection, plot_kwargs):
    # if projection == Projection.SLICE:
    line_plot_default_kwargs = {
        "color_discrete_sequence": ["#497ea7"],
        "markers": False,
    }
    scatter_plot_default_kwargs = {
        "marker": {
            "color": "#497ea7",
            "size": 10,
        }
    }

    # Update line plot kwargs if present
    if "line_plot" in plot_kwargs:
        plot_kwargs["line_plot"] = update_nested_dict(
            line_plot_default_kwargs, plot_kwargs["line_plot"]
        )
    else:
        plot_kwargs["line_plot"] = line_plot_default_kwargs

    # Update scatter plot kwargs if present
    if "scatter_plot" in plot_kwargs:
        plot_kwargs["scatter_plot"] = update_nested_dict(
            scatter_plot_default_kwargs, plot_kwargs["scatter_plot"]
        )
    else:
        plot_kwargs["scatter_plot"] = scatter_plot_default_kwargs

    if projection == Projection.SURFACE:
        surface_plot_default_kwargs = {
            "colorscale": "Blues",
            "showscale": False,
            "opacity": 0.8,
            # "scene": dict(eye={'x':1, 'y':2, 'z':0.5})
        }

        # Update surface plot kwargs if present
        if "surface_plot" in plot_kwargs:
            plot_kwargs["surface_plot"] = update_nested_dict(
                surface_plot_default_kwargs, plot_kwargs["surface_plot"]
            )
        else:
            plot_kwargs["surface_plot"] = surface_plot_default_kwargs

    elif projection == Projection.CONTOUR:
        contour_plot_default_kwargs = {
            "colorscale": "Blues",
            "showscale": False,
            "line_smoothing": 0.85,
        }

        if "contour_plot" in plot_kwargs:
            plot_kwargs["contour_plot"] = update_nested_dict(
                contour_plot_default_kwargs, plot_kwargs["contour_plot"]
            )
        else:
            plot_kwargs["contour_plot"] = contour_plot_default_kwargs
    return plot_kwargs


def evaluate_kwargs(
        projection,
        size,
        make_subplot_kwargs,
        layout_kwargs,
        plot_kwargs,
        return_dict=False,
        template=None,
):
    plot_kwargs = get_plot_kwargs(projection, plot_kwargs)

    if projection.is_single():
        cols = make_subplot_kwargs.get("cols", 1 if size == 1 else 2)
        rows = (size + cols - 1) // cols

        shared_xaxes = make_subplot_kwargs.get("shared_xaxes", True)
        shared_yaxes = make_subplot_kwargs.get("shared_yaxes", True)

        layout_kwargs = get_layout_kwargs(
            layout_kwargs, rows, cols, return_dict=return_dict
        )
        make_subplot_kwargs = get_make_subplot_kwargs(make_subplot_kwargs,
                                                      rows, cols)
        make_subplot_kwargs["shared_xaxes"] = shared_xaxes
        make_subplot_kwargs["shared_yaxes"] = shared_yaxes
    else:
        if make_subplot_kwargs:
            for key in make_subplot_kwargs.keys():
                if key in ["rows", "cols", "shared_yaxes", "shared_xaxes"]:
                    warnings.warn(
                        f"{key} param is not allowed in plot_kwargs when "
                        f"the projection is {projection.value}."
                    )
                    del make_subplot_kwargs[key]

        cols = size if size > 2 else 1
        rows = size if size > 2 else 1

        layout_kwargs = get_layout_kwargs(
            layout_kwargs,
            rows,
            cols,
            return_dict=return_dict,
            template=template,
            single_plot=size==2
        )
        make_subplot_kwargs = get_make_subplot_kwargs(
            make_subplot_kwargs, rows, cols, projection=projection,
            single_plot=size==2
        )

    return make_subplot_kwargs, layout_kwargs, plot_kwargs


def _clean_legend_duplicates(fig):
    trace_names = set()

    def disable_legend_if_duplicate(trace):
        if trace.name in trace_names:
            # in this case the legend is a duplicate
            trace.update(showlegend=False)
        else:
            trace_names.add(trace.name)

    fig.for_each_trace(disable_legend_if_duplicate)
    return fig


def combine_plots(plots, make_subplot_kwargs, layout_kwargs, expand_yrange):
    plots = deepcopy(plots)

    # Create a subplot figure
    fig = make_subplots(**make_subplot_kwargs)
    fig.update_layout(**layout_kwargs)

    # Determine rows and cols from make_subplot_kwargs
    # rows = make_subplot_kwargs.get("rows")
    # cols = make_subplot_kwargs.get("cols")

    # Add traces
    for (row_idx, col_idx), subfig in plots.items():
        for trace in subfig.data:
            fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)

        if hasattr(subfig.layout, "xaxis") and hasattr(subfig.layout.xaxis, "title"):
            fig.update_xaxes(
                title_text=subfig.layout.xaxis.title.text,
                row=row_idx + 1,
                col=col_idx + 1,
            )
        if hasattr(subfig.layout, "yaxis") and hasattr(subfig.layout.yaxis, "title"):
            if (
                    "shared_yaxes" in make_subplot_kwargs
                    and make_subplot_kwargs["shared_yaxes"]
            ):
                if col_idx == 0:
                    fig.update_yaxes(
                        title_text=subfig.layout.yaxis.title.text,
                        row=row_idx + 1,
                        col=col_idx + 1,
                    )
            else:
                fig.update_yaxes(
                    title_text=subfig.layout.yaxis.title.text,
                    row=row_idx + 1,
                    col=col_idx + 1,
                )

    # Share y-axis ranges if needed
    if "shared_yaxes" in make_subplot_kwargs and make_subplot_kwargs["shared_yaxes"]:
        all_y = []
        for subfig in plots.values():
            for trace in subfig.data:
                if "y" in trace:
                    all_y.append(np.array(trace["y"]))
        if all_y:
            y_all = np.concatenate(all_y)
            lb, ub = np.min(y_all), np.max(y_all)
            y_range = ub - lb
            y_lower = lb - expand_yrange * y_range
            y_upper = ub + expand_yrange * y_range
            fig.update_yaxes(range=[y_lower, y_upper])

    # Share x-axis ranges if needed
    if "shared_xaxes" in make_subplot_kwargs and make_subplot_kwargs["shared_xaxes"]:
        all_x = []
        for subfig in plots.values():
            for trace in subfig.data:
                if "x" in trace:
                    all_x.append(np.array(trace["x"]))
        if all_x:
            x_all = np.concatenate(all_x)
            lb, ub = np.min(x_all), np.max(x_all)
            fig.update_xaxes(range=[lb, ub])

    # Clean duplicate legends if needed
    fig = _clean_legend_duplicates(fig)

    return fig



# Plot Data
def evaluate_func(params, func, func_kwargs):
    """Evaluate a user-defined function, handling deprecated dictionary output.

    Args:
        params: Input parameters for the function.
        func: The user-defined objective function.
        func_kwargs: Optional dictionary of keyword arguments to pass to the function

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


def process_bounds(bounds, lower_bounds, upper_bounds):
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


def select_parameter_indices(converter, internal_params, selector):
    n_params = len(internal_params.values)
    if selector is None:
        return np.arange(n_params, dtype=int)

    helper = converter.params_from_internal(np.arange(n_params))
    registry = get_registry(extended=True)
    selected = np.array(
        tree_just_flatten(selector(helper), registry=registry), dtype=int
    )

    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")
    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    return selected


def generate_param_data(internal_params, selected, n_gridpoints):
    metadata = {}
    for pos in selected:
        metadata[internal_params.names[pos]] = np.linspace(
            internal_params.lower_bounds[pos],
            internal_params.upper_bounds[pos],
            n_gridpoints,
        )
    return metadata


def evaluate_function_values(
        data, internal, params, func, converter,
        batch_evaluator, n_cores, projection=None
):
    if not projection:
        projection = Projection.SLICE

    evaluation_points = generate_evaluation_points(
        data, internal, converter, params, projection
    )
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


def generate_evaluation_points(data, internal, converter, p_names, projection):
    evaluation_points = []
    point = dict(zip(internal.names, internal.values, strict=False))

    if projection.is_single():
        x = data[p_names]

        for p_value in x:
            # updating only the parameter of interest
            point[p_names] = p_value

            values = np.array(list(point.values()))
            evaluation_points.append(converter.params_from_internal(values))
    elif projection.is_multiple():
        x_name, y_name = p_names[0], p_names[1]
        x_vals = data[x_name]
        y_vals = data[y_name]

        x, y = np.meshgrid(x_vals, y_vals)
        x_ravel = x.ravel()
        y_ravel = y.ravel()
        for a, b in zip(x_ravel, y_ravel, strict=False):
            point[x_name] = a
            point[y_name] = b
            values = np.array(list(point.values()))
            evaluation_points.append(converter.params_from_internal(values))
    return evaluation_points

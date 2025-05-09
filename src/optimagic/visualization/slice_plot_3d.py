import warnings
from copy import deepcopy
from enum import Enum
from functools import partial

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
    projection="univariate",
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
    """Generate interactive slice, contour or surface plots of a function over its
    parameters.

    Produces 2D slice plots (one parameter at a time), 2D contour plots
    (two parameters), or 3D surface plots (two parameters) of a user-supplied
    function evaluated on a grid defined by parameter bounds. Individual plots can
    be returned as a dict or combined into a single
    Plotly figure with subplots.

    Args:
        func (callable): criterion function that takes params and returns a scalar,
            PyTree, or FunctionValue object.
        params (pytree): A pytree with parameters.
        bounds (optimagic.Bounds or sequence or None): Lower and upper bounds on the
            parameters. The bounds are used to create
            a grid over which slice plots are drawn. The most general and preferred
            way to specify bounds is an `optimagic.Bounds` an object that collects
            lower, upper, soft_lower, and soft_upper bounds. The soft bounds are
            not used for slice_plots. Each bound type mirrors the structure of params.
            Check our how-to guide on bounds for examples. If params is a flat numpy
            array, you can also provide bounds via any format that is supported by
            scipy.optimize.minimize.
        func_kwargs (dict or None): Extra keywords to pass to `func` on each call.
            Default: None
        selector (callable): Function that takes params and returns a subset
            of params for which we actually want to generate the plot.
            Default: None
        n_gridpoints (int): Number of gridpoints on which the criterion function is
            evaluated. This is the number per plotted line.
            Default: 20
        projection (str or Projection): Type of plot: `"slice"` (2D slice),
            `"contour"` (2D contour), or `"surface"` (3D surface).
            Default: `"slice"`
        make_subplot_kwargs (dict or None): kwargs for `plotly.subplots.make_subplots`
            Default: None.
            Internal defaults when None:
              - rows, cols computed from a number of parameters and projection
              - start_cell='top-left', print_grid=False
              - horizontal_spacing=1/(cols*5), vertical_spacing=(1/(max(rows-1,1)))/5
              - If projection is contour or surface, `specs` grid matching types are
                 added.
        layout_kwargs (dict or None): kwargs for figure layout update. Default: None.
            Internal defaults when None:
              - width, height = 450 (single plot) or 300 × cols by 300 × rows
              - template = "plotly" (multi‐parameter) or DEFAULT PLOTLY_TEMPLATE
              - xaxis_showgrid=False, yaxis_showgrid=False
        plot_kwargs (dict or None): Nested dict of trace‐level kwargs. Default: None.
            Internal defaults when None:
              - line_plot: {'color_discrete_sequence':['#497ea7'], 'markers': False}
              - scatter_plot: {'marker':{'color':'#497ea7','size':4}}
              - surface_plot (if projection="surface"):
                    {'colorscale':'Aggrnyl','showscale':False,'opacity':0.8}
              - contour_plot (if projection="contour"):
                    {'colorscale':'Aggrnyl','showscale':False,'line_smoothing':0.85}
        param_names (dict or NoneType): Dictionary mapping old parameter names
            to new ones.
            Default: None
        expand_yrange (float): The ration by which to expand the range of the
            y-axis, such that the axis is not cropped at exactly the max of
            Criterion Value.
            Default: 0.02
        batch_evaluator (str or callable): See :ref:`batch_evaluators`.
            Default: "joblib"
        n_cores (int): Number of cores.
            Default: 1
        return_dict (bool): If True, return a dict of individual figures
            keyed by (row,col). If False, return a combined Plotly Figure.
            Default: False
        lower_bounds (sequence or None): Deprecated alias for bound lower limit.
            Default: None
        upper_bounds (sequence or None): Deprecated alias for bound upper limit.
            Default: None

    Returns:
        dict or plotly.Figure:
            If `return_dict=True`, a dict mapping subplot indices to
            Plotly Figure objects. Otherwise, a single combined Plotly Figure with
            shared axes and layout.

    """
    bounds = replace_and_warn_about_deprecated_bounds(
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        bounds=bounds,
    )
    bounds = pre_process_bounds(bounds)

    if func_kwargs is not None:
        func = partial(func, **func_kwargs)

    func_eval = func(params)

    # ==================================================================================
    # handle deprecated function output
    # ==================================================================================
    if deprecations.is_dict_output(func_eval):
        msg = (
            "Functions that return dictionaries are deprecated in slice_plot and will "
            "raise an error in version 0.6.0. Please pass a function that returns a "
            "FunctionValue object instead and use the `mark` decorators to specify "
            "whether it is a scalar, least-squares or likelihood function."
        )
        warnings.warn(msg, FutureWarning)
        func_eval = deprecations.convert_dict_to_function_value(func_eval)
        func = deprecations.replace_dict_output(func)

    # ==================================================================================
    # Infer the function type and enforce the return type
    # ==================================================================================

    if deprecations.is_dict_output(func_eval):
        problem_type = deprecations.infer_problem_type_from_dict_output(func_eval)
    else:
        problem_type = infer_aggregation_level(func)

    func_eval = convert_fun_output_to_function_value(func_eval, problem_type)

    func = enforce_return_type(problem_type)(func)

    # ==================================================================================

    converter, internal_params = get_converter(
        params=params,
        constraints=None,
        bounds=bounds,
        func_eval=func_eval,
        solver_type="value",
    )

    n_params = len(internal_params.values)

    selected = np.arange(n_params, dtype=int)
    if selector is not None:
        helper = converter.params_from_internal(selected)
        registry = get_registry(extended=True)
        selected = np.array(
            tree_just_flatten(selector(helper), registry=registry), dtype=int
        )

    if not np.isfinite(internal_params.lower_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite lower bounds.")

    if not np.isfinite(internal_params.upper_bounds[selected]).all():
        raise ValueError("All selected parameters must have finite upper bounds.")

    param_data = {}
    titles = []
    for pos in selected:
        p_name = internal_params.names[pos]
        param_data[p_name] = np.linspace(
            internal_params.lower_bounds[pos],
            internal_params.upper_bounds[pos],
            n_gridpoints,
        )

        if param_names:
            titles.append(param_names.get(p_name, p_name))
        else:
            titles.append(p_name)

    projection = Projection.parse(projection)
    template = "plotly" if not is_univariate(projection) else PLOTLY_TEMPLATE

    selected_size = len(selected)

    if not is_univariate(projection) and selected_size < 2:
        raise ValueError(
            f"{projection!r} requires at least two parameters. Got {selected_size}."
        )

    plot_kwargs = evaluate_plot_kwargs(plot_kwargs)
    make_subplot_kwargs = evaluate_make_subplot_kwargs(
        make_subplot_kwargs, selected_size, projection, titles
    )
    layout_kwargs = evaluate_layout_kwargs(
        layout_kwargs,
        projection,
        subplots=make_subplot_kwargs,
        template=template,
    )
    plots = {}
    if is_univariate(projection):
        cols = make_subplot_kwargs.get("cols", 1)
        for idx, pos in enumerate(selected):
            fig = plot_univariate(
                pos,
                param_data,
                param_names,
                internal_params,
                func,
                func_eval,
                converter,
                batch_evaluator,
                n_cores,
                expand_yrange,
                plot_kwargs,
                make_subplot_kwargs,
                layout_kwargs,
                None,
            )
            row, col = divmod(idx, cols)
            plots[(row, col)] = fig
    else:
        single_plot = True if selected_size == 2 else False
        lower_projection = projection.get("lower")
        upper_projection = projection.get("upper")

        for i, pos_x in enumerate(selected):
            for j, pos_y in enumerate(selected):
                if pos_x == pos_y and single_plot:
                    print(pos_x, pos_y)
                    pos_x, pos_y = selected
                    print(pos_x, pos_y)

                # Diagonal plot are slice plots
                if i == j and not single_plot:
                    fig = plot_univariate(
                        pos_x,
                        param_data,
                        param_names,
                        internal_params,
                        func,
                        func_eval,
                        converter,
                        batch_evaluator,
                        n_cores,
                        expand_yrange,
                        plot_kwargs,
                        make_subplot_kwargs,
                        layout_kwargs,
                        Projection.UNIVARIATE,
                    )
                else:
                    subplot_projection = None
                    if i < j and upper_projection is not None:
                        subplot_projection = upper_projection
                    elif i > j and lower_projection is not None:
                        subplot_projection = lower_projection
                    elif i == j and single_plot:
                        subplot_projection = lower_projection
                    if subplot_projection is not None:
                        print(subplot_projection)
                        fig = plot_multivariate(
                            pos_x,
                            pos_y,
                            param_data,
                            internal_params,
                            param_names,
                            func,
                            func_eval,
                            converter,
                            batch_evaluator,
                            n_cores,
                            subplot_projection,
                            n_gridpoints,
                            plot_kwargs,
                            layout_kwargs,
                        )
                    else:
                        fig = go.Figure()
                plots[(i, j)] = fig
                if single_plot:
                    break
            if single_plot:
                break

    if return_dict:
        return plots
    return combine_plots(
        plots, make_subplot_kwargs, layout_kwargs, expand_yrange, titles
    )


# Plot Data


# Helper functions
def evaluate_function_values(points, func, batch_evaluator, n_cores):
    """Batch-evaluate the user function at grid points, returning a flat list of scalar
    values."""
    batch_evaluator = process_batch_evaluator(batch_evaluator)
    results = batch_evaluator(
        func=func,
        arguments=points,
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
    """Build the list of internal parameter vectors to pass to the batch evaluator."""
    evaluation_points = []
    point = dict(zip(internal.names, internal.values, strict=False))

    if is_univariate(projection):
        x = data[p_names]

        for p_value in x:
            # updating only the parameter of interest
            point[p_names] = p_value

            values = np.array(list(point.values()))
            evaluation_points.append(converter.params_from_internal(values))
    else:
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


# Plot utils

LINE_PLOT_DEFAULT_KWARGS = {
    "color_discrete_sequence": ["#497ea7"],
    "markers": False,
}

SCATTER_PLOT_DEFAULT_KWARGS = {
    "marker": {
        "color": "red",
        "size": 5,
    }
}

SURFACE_PLOT_DEFAULT_KWARGS = {
    "colorscale": "Aggrnyl",
    "showscale": False,
    "opacity": 0.8,
}

CONTOUR_PLOT_DEFAULT_KWARGS = {
    "colorscale": "Aggrnyl",
    "showscale": True,
    "line_smoothing": 0.85,
}

DEFAULT_SCENE_CAMERA_VIEW = dict(x=2, y=2, z=0.5)


def plot_univariate(
    pos,
    param_data,
    param_names,
    internal_params,
    func,
    func_eval,
    converter,
    batch_evaluator,
    n_cores,
    expand_yrange,
    plot_kwargs,
    make_subplot_kwargs,
    layout_kwargs,
    projection=None,
):
    """Generate a 2D slice plot for a single parameter index.

    1. Extracts the parameter name and display label.
    2. Builds arrays of parameter values across a predefined grid.
    3. Evaluates the target function at each grid point in batch mode,
       leveraging parallelism as configured by `batch_evaluator` and `n_cores`.
    4. Computes an expanded y-axis range based on `expand_yrange` to ensure
       adequate plot padding.
    5. Creates a Plotly line plot of the function values and overlays a
       scatter marker at the initial parameters.

    Returns:
        go.Figure: A Plotly figure with line and marker traces, ready for
        integration into subplots or standalone display.

    """
    param_name = internal_params.names[pos]
    display_name = (
        param_names.get(param_name, param_name) if param_names else param_name
    )
    evaluation_points = generate_evaluation_points(
        param_data, internal_params, converter, param_name, Projection.UNIVARIATE
    )

    x = param_data[param_name].tolist()
    y = evaluate_function_values(evaluation_points, func, batch_evaluator, n_cores)

    y_range = compute_yaxis_range(y, expand_yrange)
    fig = plot_line(
        x=x,
        y=y,
        point={
            "x": [internal_params.values[pos]],
            "y": [func_eval.internal_value(AggregationLevel.SCALAR)],
        },
        display_name=display_name,
        y_range=y_range,
        plot_kwargs=plot_kwargs,
        make_subplot_kwargs=make_subplot_kwargs,
        layout_kwargs=layout_kwargs,
        projection=projection,
    )
    return fig


def plot_multivariate(
    pos_x,
    pos_y,
    param_data,
    internal_params,
    param_names,
    func,
    func_eval,
    converter,
    batch_evaluator,
    n_cores,
    projection,
    n_gridpoints,
    plot_kwargs,
    layout_kwargs,
):
    """Generate a 2D contour or 3D surface plot for two parameters.

    # TODO: avoid redundant function evaluations to computational efficacy

    1. Maps the selected parameter indices (`pos_x`, `pos_y`) to their names.
    2. Constructs a meshgrid of x-y values over the bounds grid.
    3. Batch-evaluates the user function at each (x,y) pair, reshaping
       results into a matrix of size (n_gridpoints, n_gridpoints).
    4. Depending on `projection`, dispatches to either `plot_contour` or
      `plot_surface`, passing through customized `plot_kwargs`.

    Returns:
        go.Figure: A contour or surface figure, optionally annotated with
        the function value at the initial parameters.

    """
    x_name = internal_params.names[pos_x]
    y_name = internal_params.names[pos_y]

    evaluation_points = generate_evaluation_points(
        param_data, internal_params, converter, [x_name, y_name], projection
    )

    x = param_data[x_name]
    y = param_data[y_name]
    z = evaluate_function_values(evaluation_points, func, batch_evaluator, n_cores)

    x, y = np.meshgrid(x, y)
    z = np.reshape(z, (n_gridpoints, n_gridpoints))
    display = {
        "x": param_names.get(x_name, x_name) if param_names else x_name,
        "y": param_names.get(y_name, y_name) if param_names else y_name,
    }
    if is_surface(projection):
        return plot_surface(
            x,
            y,
            z,
            plot_kwargs=plot_kwargs,
            layout_kwargs=layout_kwargs,
            point={
                "x": [internal_params.values[pos_x]],
                "y": [internal_params.values[pos_y]],
                "z": [func_eval.internal_value(AggregationLevel.SCALAR)],
            },
        )
    return plot_contour(
        x,
        y,
        z,
        plot_kwargs=plot_kwargs,
        layout_kwargs=layout_kwargs,
        point={
            "x": [internal_params.values[pos_x]],
            "y": [internal_params.values[pos_y]],
        },
    )


def plot_line(
    x,
    y,
    y_range=None,
    point=None,
    display_name=None,
    plot_kwargs=None,
    make_subplot_kwargs=None,
    layout_kwargs=None,
    projection=None,
):
    """Create a 2D line plot with an initial parameter marker.

    1. Uses Plotly Express to draw a line of y vs. x with settings from
      `plot_kwargs['line_plot']`.
    2. If `point` is provided, overlays a scatter trace at the specified
      (x,y) coordinates using `plot_kwargs['scatter_plot']`.
    3. Applies axis titles: `display_name` on x-axis, "Function Value" on y-axis.
    4. Honors `shared_yaxes` from `make_subplot_kwargs` to set a common y-range.
    5. Updates figure layout from `layout_kwargs`.

    Returns:
        go.Figure: Configured 2D line and marker plot.

    """
    fig = px.line(x=x, y=y, **plot_kwargs["line_plot"])

    if point:
        fig.add_trace(
            go.Scatter(x=point["x"], y=point["y"], **plot_kwargs["scatter_plot"])
        )

    if layout_kwargs:
        fig.update_layout(**layout_kwargs)

    if not is_univariate(projection):
        fig.update_xaxes(title={"text": display_name})
        fig.update_yaxes(
            title={"text": "Function Value"},
            range=y_range
            if "shared_yaxes" in make_subplot_kwargs
            and make_subplot_kwargs["shared_yaxes"]
            else None,
        )
    else:
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
    return fig


def plot_surface(x, y, z, layout_kwargs=None, plot_kwargs=None, point=None):
    """Construct a 3D surface plot of z = f(x,y,z).

    1. Creates a `go.Surface` trace with x, y, z matrices and
      styling from `plot_kwargs['surface_plot']`.
    2. Builds a `go.Figure` with the surface as its primary data.
    3. If `point` is specified, adds a 3D scatter marker at the
      current (x,y,z) location.
    4. Applies overall scene and layout configurations via `layout_kwargs`.

    Returns:
        go.Figure: A 3D surface visualization.

    """
    trace = go.Surface(
        z=z, x=x, y=y, **plot_kwargs["surface_plot"], coloraxis="coloraxis"
    )

    fig = go.Figure(data=[trace], layout=layout_kwargs)

    if point:
        fig.add_trace(
            go.Scatter3d(
                x=point["x"], y=point["y"], z=point["z"], **plot_kwargs["scatter_plot"]
            )
        )
    return fig


def plot_contour(x, y, z, layout_kwargs=None, plot_kwargs=None, point=None):
    """Build a 2D contour plot of z = f(x,y,z).

    1. Uses `go.Contour` with flattened x and y axes and z data,
      styled via `plot_kwargs['contour_plot']`.
    2. Wraps the contour in a `go.Figure` and applies `layout_kwargs`.
    3. Optionally overlays a 3D scatter marker for the current parameter
      point (projected onto the x-y plane).

    Returns:
        go.Figure: A contour map with optional annotation.

    """
    trace = go.Contour(
        z=z, x=x[0], y=y[:, 0], **plot_kwargs["contour_plot"], coloraxis="coloraxis"
    )

    fig = go.Figure(data=[trace], layout=layout_kwargs)
    if point:
        fig.add_trace(
            go.Scatter(x=point["x"], y=point["y"], **plot_kwargs["scatter_plot"]),
        )

    return fig


# Plot utils
class Projection(str, Enum):
    UNIVARIATE = "univariate"
    CONTOUR = "contour"
    SURFACE = "surface"

    @classmethod
    def parse(cls, value):
        def validate_projection(val):
            if val is None:
                return None
            if isinstance(val, str):
                val = val.lower()
                if val in (cls.SURFACE, cls.CONTOUR):
                    return val
                raise ValueError(f"Invalid projection: '{val}'")
            raise TypeError(f"Expected str or None in dict values, got {type(val)}")

        if isinstance(value, str):
            val = value.lower()
            if val == cls.UNIVARIATE:
                return cls.UNIVARIATE
            elif val in (cls.SURFACE, cls.CONTOUR):
                return {"lower": val, "upper": None}
            else:
                raise ValueError(f"Invalid projection: '{val}'")

        if isinstance(value, dict):
            lower = validate_projection(value.get("lower"))
            upper = validate_projection(value.get("upper"))
            return {"lower": lower, "upper": upper}

        raise TypeError(
            f"Invalid type for projection: {type(value)}, "
            f"only str ('univariate', 'surface', 'contour') "
            f"or dict allowed with 'lower' and 'upper' keys."
        )


def compute_yaxis_range(y, expand_yrange):
    # Calculate expanded y-axis limits based on data range
    y_min, y_max = np.min(y), np.max(y)
    y_range = y_max - y_min
    return [y_min - expand_yrange * y_range, y_max + expand_yrange * y_range]


def is_univariate(value):
    return value == Projection.UNIVARIATE


def is_surface(value):
    return value == Projection.SURFACE


def is_contour(value):
    return value == Projection.CONTOUR


def _clean_legend_duplicates(fig):
    """Remove duplicate legend entries from a combined Plotly figure."""
    trace_names = set()
    dup_names = set()

    def disable_legend_if_duplicate(trace):
        print(trace.type)

        if trace.type == "contour" and trace.name in dup_names:
            trace.update(showscale=False)
        else:
            dup_names.add(trace.name)

        if trace.name in trace_names:
            # in this case the legend is a duplicate
            trace.update(showlegend=False)
        else:
            trace_names.add(trace.name)

    fig.for_each_trace(disable_legend_if_duplicate)
    return fig


def combine_plots(
    plots, make_subplot_kwargs, layout_kwargs, expand_yrange, titles=None
):
    """Combine individual subplot figures into one Plotly Figure, sharing axes and
    layout."""
    plots = deepcopy(plots)
    titles = make_subplot_kwargs["row_titles"]
    if make_subplot_kwargs["rows"] == 1 and make_subplot_kwargs["cols"] == 1:
        make_subplot_kwargs["row_titles"] = [titles[0]]
        make_subplot_kwargs["column_titles"] = [titles[1]]

    print(make_subplot_kwargs)

    # Create a subplot figure
    fig = make_subplots(**make_subplot_kwargs)
    fig.update_layout(**layout_kwargs)

    fig.for_each_annotation(
        lambda a: a.update(y=-0.07)
        if abs(a["y"] - 1) < 1e-3
        else a.update(x=-0.07, textangle=270)
        if abs(a["x"] - 0.98) < 1e-3
        else None
    )

    # Add traces
    for (row_idx, col_idx), subfig in plots.items():
        for trace in subfig.data:
            fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)
            print("Trace: ", trace)
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


def evaluate_plot_kwargs(plot_kwargs):
    """Generate a default set of Plotly plot kwargs for individual plots. Merges user-
    supplied `plot_kwargs` if provided, overriding defaults.

    Returns:
        dict: kwargs for individual subpl   ot(s)

    """

    def update_nested_dict(default, updates):
        # Recursively merge `updates` into `default`
        for k, v in updates.items():
            if isinstance(v, dict) and k in default and isinstance(default[k], dict):
                default[k] = update_nested_dict(default[k], v)
            else:
                default[k] = v
        return default

    if plot_kwargs is None:
        plot_kwargs = {}

    # Define default settings
    defaults = {
        "line_plot": LINE_PLOT_DEFAULT_KWARGS,
        "scatter_plot": SCATTER_PLOT_DEFAULT_KWARGS,
        "surface_plot": SURFACE_PLOT_DEFAULT_KWARGS,
        "contour_plot": CONTOUR_PLOT_DEFAULT_KWARGS,
    }

    plot_defaults = {}
    for key in defaults.keys():
        user_kwargs = plot_kwargs.get(key, {})
        plot_defaults[key] = update_nested_dict(deepcopy(defaults[key]), user_kwargs)

    return plot_defaults


def evaluate_make_subplot_kwargs(make_subplot_kwargs, size, projection, titles):
    """Assemble default kwargs for `plotly.subplots.make_subplots`. User-supplied
    `make_subplot_kwargs` override these defaults.

    Returns:
        dict: Kwargs for `make_subplots()`

    """
    if make_subplot_kwargs is None:
        make_subplot_kwargs = {}
    make_subplot_defaults = {}

    if make_subplot_kwargs and not is_univariate(projection):
        for key in make_subplot_kwargs.keys():
            if key in ["rows", "cols"]:
                raise ValueError(
                    f"{key} param is not allowed in plot_kwargs when "
                    f"the projection is {projection}."
                )

    if is_univariate(projection):
        cols = make_subplot_kwargs.get("cols", 1 if size == 1 else 2)
        rows = (size + cols - 1) // cols

        make_subplot_defaults["shared_xaxes"] = True
        make_subplot_defaults["shared_yaxes"] = True
    else:
        cols = size if size > 2 else 1
        rows = size if size > 2 else 1

        if size > 2:
            specs = []
            for i in range(rows):
                row_specs = []
                for j in range(cols):
                    if i == j:
                        cell_type = "xy"
                    elif i > j:
                        cell_type = projection.get("lower")
                    else:
                        cell_type = projection.get("upper")

                    if cell_type is not None:
                        if cell_type == "xy":
                            row_specs.append({"type": "xy"})
                        elif is_surface(cell_type):
                            row_specs.append({"type": "scene"})
                        elif is_contour(cell_type):
                            row_specs.append({"type": "contour"})
                    else:
                        row_specs.append({})
                specs.append(row_specs)
        else:
            if is_surface(projection.get("lower")):
                specs = [[{"type": "scene"}]]
            else:
                specs = [[{"type": "contour"}]]

        make_subplot_defaults["specs"] = specs
        make_subplot_defaults["row_titles"] = titles
        make_subplot_defaults["column_titles"] = titles

    print("titles: ", make_subplot_defaults["row_titles"])
    make_subplot_defaults.update(
        {
            "rows": rows,
            "cols": cols,
            "horizontal_spacing": 1 / (cols * 5),
            "vertical_spacing": (1 / max(rows - 1, 1)) / 5,
        }
    )

    make_subplot_defaults.update(make_subplot_kwargs)

    return make_subplot_defaults


def evaluate_layout_kwargs(
    layout_kwargs, projection, subplots=None, template=PLOTLY_TEMPLATE
):
    """Generate a default set of Plotly layout kwargs for subplots. Merges user-supplied
    `layout_kwargs` if provided, overriding defaults.

    Returns:
        dict: kwargs for `Figure.update_layout()`

    """
    if layout_kwargs is None:
        layout_kwargs = {}
    layout_defaults = {}

    if subplots is not None and (subplots.get("rows") > 1 or subplots.get("cols") > 1):
        width = 300 * subplots.get("cols")
        height = 300 * subplots.get("rows")
    else:
        width = 450
        height = 450

    if not is_univariate(projection):
        eye_layouts = {}
        scene_counter = 0

        rows = subplots.get("rows")
        cols = subplots.get("cols")

        eye_layouts["coloraxis"] = {"colorscale": "aggrnyl"}

        if "specs" in subplots:
            specs = subplots["specs"]
            for i in range(rows):
                for j in range(cols):
                    if "type" in specs[i][j] and specs[i][j]["type"] == "scene":
                        scene_counter += 1
                        scene_id = f"scene{scene_counter}"
                        eye_layouts[f"{scene_id}"] = {
                            "camera": {"eye": DEFAULT_SCENE_CAMERA_VIEW},
                            "xaxis": dict(title="", nticks=4),
                            "yaxis": dict(title="", nticks=4),
                            "zaxis": dict(title="", nticks=4),
                        }

            layout_defaults.update(eye_layouts)

    layout_defaults.update(
        {
            "width": width,
            "height": height,
            "template": template,
            "showlegend": False,
        }
    )

    layout_defaults.update(layout_kwargs)
    return layout_defaults
